from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float
    
    def __post_init__(self):
        """初始化后验证宽度和高度是否为正数"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive values.")
        self._width = self.width  # 初始化 _width
        self._height = self.height  # 初始化 _height

    @property
    def width(self):
        """获取宽度"""
        return self._width
    
    @width.setter
    def width(self, value):
        """设置宽度，并确保宽度为正数"""
        if value <= 0:
            raise ValueError("Width must be a positive number.")
        self._width = value

    @property
    def height(self):
        """获取高度"""
        return self._height
    
    @height.setter
    def height(self, value):
        """设置高度，并确保高度为正数"""
        if value <= 0:
            raise ValueError("Height must be a positive number.")
        self._height = value

    @property
    def area(self):
        """计算矩形的面积，面积是只读的"""
        return self.width * self.height

    @property
    def perimeter(self):
        """计算矩形的周长，周长是只读的"""
        return 2 * (self.width + self.height)

    @property
    def is_square(self):
        """判断矩形是否是正方形"""
        return self.width == self.height

    def __str__(self):
        """返回矩形的字符串表示"""
        return f"Rectangle(width={self.width}, height={self.height})"

    def scale(self, factor: float):
        """按比例缩放矩形
        
        Args:
            factor: 缩放因子，必须为正数
        """
        if factor <= 0:
            raise ValueError("Scaling factor must be positive")
        self.width *= factor
        self.height *= factor


def main():
    try:
        # 创建一个矩形实例
        rect = Rectangle(width=5, height=3)
        print(f"创建矩形: {rect}")
        
        # 测试基本属性
        print(f"面积: {rect.area}")
        print(f"周长: {rect.perimeter}")
        print(f"是否为正方形: {rect.is_square}")
        
        # 测试缩放功能
        rect.scale(2)
        print(f"\n放大2倍后:")
        print(f"新尺寸: {rect}")
        print(f"新面积: {rect.area}")
        
        # 测试正方形情况
        square = Rectangle(width=4, height=4)
        print(f"\n创建正方形: {square}")
        print(f"是否为正方形: {square.is_square}")
        
        # 测试异常处理
        try:
            invalid_rect = Rectangle(width=-1, height=5)
        except ValueError as e:
            print(f"\n测试负数宽度: {e}")
            
        try:
            rect.height = 0
        except ValueError as e:
            print(f"测试零高度: {e}")
            
    except Exception as e:
        print(f"发生错误: {e}")
        

if __name__ == "__main__":
    main()

