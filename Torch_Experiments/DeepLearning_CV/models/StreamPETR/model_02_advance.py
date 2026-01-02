import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =========================================================
# Utils
# =========================================================
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def normalize_pc_range(xyz: torch.Tensor, pc_range) -> torch.Tensor:
    """
    xyz: (..., 3) in ego/world coords
    pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
    returns: (..., 3) in [0,1]
    """
    device, dtype = xyz.device, xyz.dtype
    mins = torch.tensor(pc_range[:3], device=device, dtype=dtype)
    maxs = torch.tensor(pc_range[3:], device=device, dtype=dtype)
    return (xyz - mins) / (maxs - mins)


def denormalize_pc_range(xyz01: torch.Tensor, pc_range) -> torch.Tensor:
    device, dtype = xyz01.device, xyz01.dtype
    mins = torch.tensor(pc_range[:3], device=device, dtype=dtype)
    maxs = torch.tensor(pc_range[3:], device=device, dtype=dtype)
    return xyz01 * (maxs - mins) + mins


# =========================================================
# Config
# =========================================================
@dataclass
class StreamPETRConfig:
    # image
    num_cams: int = 6
    img_h: int = 256
    img_w: int = 704
    stride: int = 16

    # dims
    embed_dim: int = 256
    num_heads: int = 8
    num_decoder_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1

    # depth bins (PETR frustum discretization)
    num_depth: int = 4
    depth_values: Tuple[float, ...] = (10.0, 20.0, 30.0, 40.0)

    # 3D range (ego frame)
    pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)

    # queries
    num_new_queries: int = 256      # new queries injected each frame
    topk_per_frame: int = 256       # how many queries to push into memory each frame

    # temporal memory queue
    mem_frames: int = 4             # keep last N frames
    # total mem queries = mem_frames * topk_per_frame

    # heads
    num_classes: int = 10
    box_dim: int = 10  # [cx,cy,cz] + other 7 dims (w,l,h,sin,cos,vx,vy) for example


# =========================================================
# Backbone (mock)
# =========================================================
class SimpleBackbone(nn.Module):
    """
    Input:  (B*N, 3, H, W)
    Output: (B*N, C, H/stride, W/stride)
    """
    def __init__(self, cfg: StreamPETRConfig):
        super().__init__()
        C = cfg.embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(inplace=True),    # /2
            nn.MaxPool2d(3, 2, 1),                                 # /4
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),   # /8
            nn.Conv2d(128, C, 3, 2, 1), nn.ReLU(inplace=True),    # /16
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Motion-aware LayerNorm (simplified MLN)
# =========================================================
class MotionAwareLN(nn.Module):
    """
    Simplified MLN:
      gamma,beta are predicted from (ego_motion, dt).
      then apply: y = gamma * LN(x) + beta
    """
    def __init__(self, embed_dim: int, cond_dim: int = 17):  # 16 (4x4) + 1 dt
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, M, C)
        cond: (B, M, cond_dim) or (B, 1, cond_dim) broadcastable
        """
        y = self.ln(x)
        gb = self.mlp(cond)  # (B,M,2C)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1 + gamma) * y + beta  # (B,M,C)


# =========================================================
# StreamPETR-lite
# =========================================================
class StreamPETR(nn.Module):
    """
    I/O shapes (per frame):
      imgs:   (B, Ncam, 3, H, W)
      rots:   (B, Ncam, 3, 3)  cam->ego rotation  (nuScenes calibrated_sensor)
      trans:  (B, Ncam, 3)     cam->ego translation
      intrins:(B, Ncam, 3, 3)
      ego_motion_prev2cur: (B,4,4)  (optional) transform points from prev ego to cur ego

    State (returned / fed to next frame):
      state["mem_queries"]: list/deque of length <= mem_frames,
           each item: dict with:
             "q":   (B, K, C)
             "ref": (B, K, 3)  normalized [0,1] center
             "T":   (B, 4, 4)  ego_motion for that frame->current? (stored as needed)
             "dt":  (B, 1)     time gap for that frame to next
    """
    def __init__(self, cfg: StreamPETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleBackbone(cfg)

        # feature map size
        self.feat_h = cfg.img_h // cfg.stride
        self.feat_w = cfg.img_w // cfg.stride
        self.HW = self.feat_h * self.feat_w

        # pixel-center grid in image coords (u,v)
        ys = torch.arange(self.feat_h, dtype=torch.float32) + 0.5
        xs = torch.arange(self.feat_w, dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        uu = uu * cfg.stride
        vv = vv * cfg.stride
        grid_uv = torch.stack([uu, vv], dim=-1)  # (Hf,Wf,2)
        self.register_buffer("grid_uv", grid_uv, persistent=False)

        depth = torch.tensor(cfg.depth_values, dtype=torch.float32).view(1, 1, 1, cfg.num_depth)  # (1,1,1,D)
        self.register_buffer("depth_values", depth, persistent=False)

        # image 3D position encoder: (3D)->C
        self.img_pe_encoder = nn.Sequential(
            nn.Linear(3 * cfg.num_depth, cfg.embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )

        # query embeddings (new queries each frame)
        self.new_query_embed = nn.Embedding(cfg.num_new_queries, cfg.embed_dim)
        self.new_ref_points = nn.Embedding(cfg.num_new_queries, 3)  # learnable in logit space effectively

        # query position encoding from ref points
        self.query_pe_encoder = nn.Sequential(
            nn.Linear(3, cfg.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        # MLN
        self.mln = MotionAwareLN(cfg.embed_dim, cond_dim=17)

        # transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=False,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)

        # heads
        self.cls_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.reg_head = nn.Linear(cfg.embed_dim, cfg.box_dim)

        # small LN for stability
        self.mem_ln = nn.LayerNorm(cfg.embed_dim)

    # ---------------------------
    # PETR backprojection to ego
    # ---------------------------
    def backproject_xyz_ego(
        self,
        rots: torch.Tensor,
        trans: torch.Tensor,
        intrins: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns xyz_ego for each cam token and each depth bin.
        xyz_ego: (B, Ncam, HW, D, 3)
        """
        B, N = trans.shape[:2]
        HW, D = self.HW, self.cfg.num_depth
        Hf, Wf = self.feat_h, self.feat_w

        uv = self.grid_uv.to(device=trans.device, dtype=trans.dtype)  # (Hf,Wf,2)
        ones = torch.ones((Hf, Wf, 1), device=trans.device, dtype=trans.dtype)
        uv1 = torch.cat([uv, ones], dim=-1).view(1, 1, HW, 3).repeat(B, N, 1, 1)  # (B,N,HW,3)

        K_inv = torch.linalg.inv(intrins)  # (B,N,3,3)
        ray = (K_inv.unsqueeze(2) @ uv1.unsqueeze(-1)).squeeze(-1)  # (B,N,HW,3)

        ray_d = ray.unsqueeze(3) * self.depth_values.to(ray).unsqueeze(-1)  # (B,N,HW,D,3)

        R = rots.view(B, N, 1, 1, 3, 3)
        t = trans.view(B, N, 1, 1, 3)
        xyz = (R @ ray_d.unsqueeze(-1)).squeeze(-1) + t  # (B,N,HW,D,3)
        return xyz

    def build_img_memory(
        self,
        imgs: torch.Tensor,
        rots: torch.Tensor,
        trans: torch.Tensor,
        intrins: torch.Tensor
    ) -> torch.Tensor:
        """
        Build image tokens (memory) with PETR-style 3D PE.
        Returns memory: (S=Ncam*HW, B, C)
        """
        B, Ncam, _, H, W = imgs.shape
        x = imgs.view(B * Ncam, 3, H, W)
        feat = self.backbone(x)  # (B*Ncam, C, Hf, Wf)
        C = feat.shape[1]
        feat = feat.view(B, Ncam, C, self.feat_h, self.feat_w)
        feat_tok = feat.flatten(3).permute(0, 1, 3, 2).contiguous()  # (B,Ncam,HW,C)
        feat_tok = self.mem_ln(feat_tok)

        xyz = self.backproject_xyz_ego(rots, trans, intrins)  # (B,Ncam,HW,D,3)
        xyz01 = normalize_pc_range(xyz, self.cfg.pc_range).clamp(0.0, 1.0)
        xyz_pe = inverse_sigmoid(xyz01)                       # (B,Ncam,HW,D,3)
        xyz_pe = xyz_pe.permute(0, 1, 2, 4, 3).contiguous().view(B, Ncam, self.HW, 3 * self.cfg.num_depth)

        pos = self.img_pe_encoder(xyz_pe)  # (B,Ncam,HW,C)
        tok = feat_tok + pos               # (B,Ncam,HW,C)

        memory = tok.view(B, Ncam * self.HW, C).permute(1, 0, 2).contiguous()
        return memory

    # ---------------------------
    # motion compensation for ref points
    # ---------------------------
    def motion_compensate_ref(
        self,
        prev_ref01: torch.Tensor,
        ego_motion_prev2cur: torch.Tensor
    ) -> torch.Tensor:
        """
        这一步并没有处理任何“特征”，只是把“坐标”修正了。
        比如上一帧车在 (10, 0)，我往前开了 2 米，这个函数算出来新坐标是 (8, 0)
        
        prev_ref01: (B,M,3) in [0,1], ego frame prev
        ego_motion_prev2cur: (B,4,4) transform points prev_ego -> cur_ego
        returns: (B,M,3) in [0,1]
        """
        prev_xyz = denormalize_pc_range(prev_ref01, self.cfg.pc_range)  # (B,M,3)
        B, M, _ = prev_xyz.shape
        ones = torch.ones(B, M, 1, device=prev_xyz.device, dtype=prev_xyz.dtype)
        prev_h = torch.cat([prev_xyz, ones], dim=-1)  # (B,M,4)

        cur_h = (ego_motion_prev2cur @ prev_h.transpose(1, 2)).transpose(1, 2)  # (B,M,4)
        cur_xyz = cur_h[..., :3]
        cur01 = normalize_pc_range(cur_xyz, self.cfg.pc_range).clamp(0.0, 1.0)
        return cur01

    # ---------------------------
    # select topK to push into memory
    # ---------------------------
    def select_topk(self, logits: torch.Tensor, center01: torch.Tensor, q_content: torch.Tensor, K: int):
        """
        logits:  (B, N_all, num_classes)
        center01:(B, N_all, 3)
        q_content:(B,N_all,C)
        """
        probs = logits.sigmoid()
        scores = probs.max(dim=-1).values  # (B,N_all)
        topk_scores, topk_idx = scores.topk(K, dim=1)

        B = logits.size(0)
        b = torch.arange(B, device=logits.device)[:, None]

        q = q_content[b, topk_idx].detach()
        ref = center01[b, topk_idx].detach()
        return q, ref

    # ---------------------------
    # forward (one frame)
    # ---------------------------
    def forward(
        self,
        imgs: torch.Tensor,
        rots: torch.Tensor,
        trans: torch.Tensor,
        intrins: torch.Tensor,
        ego_motion_prev2cur: Optional[torch.Tensor] = None,
        dt_prev2cur: Optional[torch.Tensor] = None,
        state: Optional[Dict] = None,
    ):
        """
        Returns:
          pred_logits: (B, N_all, num_classes)
          pred_boxes:  (B, N_all, box_dim)  where first 3 are center01 in [0,1]
          next_state:  dict with updated memory queue
        """
        cfg = self.cfg
        B = imgs.size(0)
        device = imgs.device

        # 1) build image memory tokens
        memory = self.build_img_memory(imgs, rots, trans, intrins)  # (S,B,C)

        # 2) prepare new queries
        new_q = self.new_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,Qnew,C)
        new_ref01 = self.new_ref_points.weight.unsqueeze(0).repeat(B, 1, 1).sigmoid()  # (B,Qnew,3)

        # 3) load memory queue -> previous queries (flatten frames)
        mem_q_list = []
        mem_ref_list = []
        if state is not None and "mem_queue" in state and len(state["mem_queue"]) > 0:
            
            # state["mem_queue"] is deque of dicts: {"q":(B,K,C), "ref":(B,K,3)}
            for item in state["mem_queue"]:
                mem_q_list.append(item["q"])
                mem_ref_list.append(item["ref"])

        if len(mem_q_list) > 0:
            prev_q = torch.cat(mem_q_list, dim=1)      # (B, N*K, C)
            prev_ref01 = torch.cat(mem_ref_list, dim=1) # (B, N*K, 3)

            # motion compensation for ref points (only if ego_motion provided)
            if ego_motion_prev2cur is None:
                # if not provided, assume identity (no ego motion)
                ego_motion_prev2cur = torch.eye(4, device=device, dtype=imgs.dtype).view(1, 4, 4).repeat(B, 1, 1)

            aligned_ref01 = self.motion_compensate_ref(prev_ref01, ego_motion_prev2cur)  # (B,NK,3)

            # MLN condition (ego motion + dt)
            if dt_prev2cur is None:
                dt_prev2cur = torch.ones(B, 1, device=device, dtype=imgs.dtype)  # dummy 1

            # build cond per token: (B, NK, 17) = [flatten(ego_motion 16), dt]
            Tflat = ego_motion_prev2cur.reshape(B, 1, 16).repeat(1, aligned_ref01.size(1), 1)
            dt = dt_prev2cur.view(B, 1, 1).repeat(1, aligned_ref01.size(1), 1)
            cond = torch.cat([Tflat, dt], dim=-1)  # (B,NK,17)

            prev_q = self.mln(prev_q, cond)  # (B,NK,C)

            # concat old + new
            active_q = torch.cat([prev_q, new_q], dim=1)                 # (B, NK+Qnew, C)
            active_ref01 = torch.cat([aligned_ref01, new_ref01], dim=1)  # (B, NK+Qnew, 3)
        else:
            active_q = new_q
            active_ref01 = new_ref01

        N_all = active_q.size(1)

        # 4) query positional encoding from ref points (use logit space)
        query_pos_emb = self.query_pe_encoder(inverse_sigmoid(active_ref01))  # (B,N_all,C)

        # tgt for decoder: (S_tgt, B, C)
        tgt = (active_q + query_pos_emb).permute(1, 0, 2).contiguous()

        # 5) transformer decode (tgt self-attn ~= hybrid interaction, then cross-attn to memory)
        hs = self.decoder(tgt=tgt, memory=memory)  # (N_all, B, C)
        hs = hs.permute(1, 0, 2).contiguous()      # (B, N_all, C)

        # 6) heads
        pred_logits = self.cls_head(hs)             # (B, N_all, num_classes)
        delta = self.reg_head(hs)                   # (B, N_all, box_dim)

        # correct center parameterization: center01 in [0,1]
        center_logit = inverse_sigmoid(active_ref01) + delta[..., :3]
        center01 = center_logit.sigmoid()

        # DO NOT sigmoid all dims
        pred_boxes = torch.cat([center01, delta[..., 3:]], dim=-1)  # (B, N_all, box_dim)

        # 7) push topK into memory queue
        push_q, push_ref01 = self.select_topk(pred_logits, center01, hs, cfg.topk_per_frame)

        if state is None or "mem_queue" not in state:
            mem_queue = deque([], maxlen=cfg.mem_frames)
        else:
            mem_queue = state["mem_queue"]

        mem_queue.append({
            "q": push_q,         # (B,K,C)
            "ref": push_ref01,   # (B,K,3)
        })

        next_state = {"mem_queue": mem_queue}
        return pred_logits, pred_boxes, next_state


# =========================================================
# Demo (2-frame streaming)
# =========================================================
def demo():
    cfg = StreamPETRConfig()
    model = StreamPETR(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    B = 1
    imgs0 = torch.randn(B, cfg.num_cams, 3, cfg.img_h, cfg.img_w, device=device)

    # nuScenes-style camera params (here mocked)
    rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, cfg.num_cams, 1, 1)   # cam->ego
    trans = torch.zeros(B, cfg.num_cams, 3, device=device)
    intrins = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, cfg.num_cams, 1, 1)
    intrins[:, :, 0, 0] = 500.0
    intrins[:, :, 1, 1] = 500.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0

    state = None
    with torch.no_grad():
        logits0, boxes0, state = model(imgs0, rots, trans, intrins, state=state)
    print("Frame0 logits:", logits0.shape, "boxes:", boxes0.shape, "mem_frames:", len(state["mem_queue"]))

    # Frame1 with ego motion (prev->cur)
    imgs1 = torch.randn(B, cfg.num_cams, 3, cfg.img_h, cfg.img_w, device=device)
    ego_motion = torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
    # example: move forward in ego-x by +1m (this depends on your ego axis convention!)
    ego_motion[:, 0, 3] = 1.0
    dt = torch.ones(B, 1, device=device) * 0.5  # 0.5s

    with torch.no_grad():
        logits1, boxes1, state = model(
            imgs1, rots, trans, intrins,
            ego_motion_prev2cur=ego_motion,
            dt_prev2cur=dt,
            state=state
        )
    print("Frame1 logits:", logits1.shape, "boxes:", boxes1.shape, "mem_frames:", len(state["mem_queue"]))
    print("Total mem queries stored:", len(state["mem_queue"]) * cfg.topk_per_frame)


if __name__ == "__main__":
    demo()
