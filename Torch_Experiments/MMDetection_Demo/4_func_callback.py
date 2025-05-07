class Trainer:
    """模拟一个训练流程，支持在每个 epoch 结束时调用回调。"""
    def __init__(self, epochs):
        self.epochs = epochs
        self.callbacks = []

    def register_callback(self, fn):
        """把用户的回调函数加入列表。"""
        self.callbacks.append(fn)

    def fit(self):
        """主流程：跑 epochs 并在每个 epoch 结束后触发所有回调。"""
        for epoch in range(1, self.epochs + 1):
            
            loss = 1.0 / epoch
            print(f"Epoch {epoch} training... loss={loss:.4f}")

            # 在这里会调用回调函数
            for cb in self.callbacks:
                cb(epoch, loss)


def on_epoch_end(epoch, loss):
    """被 Trainer 在每个 epoch 结束时调用。"""
    if loss < 0.3:
        print(f"  🎉 Epoch {epoch}: loss below threshold! ({loss:.4f})")
    else:
        print(f"  ➡️  Epoch {epoch}: continue training.")


# =============================================================
# 没有回调函数，只能在for loop 里面进行调用和修改
def log_loss(epoch, loss):
    print(f"[Manual] Epoch {epoch}: loss={loss:.4f}")

def train_loop(epochs):
    for epoch in range(1, epochs+1):
        
        # …训练逻辑…
        loss = 1.0 / epoch
        log_loss(epoch, loss)   # ★ 只能在这里写死调用
# =============================================================


if __name__ == "__main__":
    trainer = Trainer(epochs=5)

    # 注册回调：当每个 epoch 结束时，都会调用 on_epoch_end()
    trainer.register_callback(on_epoch_end)

    # 启动训练
    trainer.fit()

