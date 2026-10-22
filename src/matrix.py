from __future__ import annotations
from fractions import Fraction
from typing import List, Union


class Matrix:
    """矩阵类：使用二维列表存储元素，支持精确分数运算。"""

    def __init__(self, data: List[List[Union[int, float, Fraction]]]):
        if not data or not data[0]:
            raise ValueError("矩阵不能为空")
        rows = len(data)
        cols = len(data[0])
        for i, row in enumerate(data):
            if len(row) != cols:
                raise ValueError(f"第 {i} 行长度 ({len(row)}) 与第 0 行 ({cols}) 不一致")
        self.rows: int = rows
        self.cols: int = cols
        self.data: List[List[Fraction]] = [
            [Fraction(x) if not isinstance(x, Fraction) else x for x in row]
            for row in data
        ]

    def copy(self) -> Matrix:
        return Matrix([[self.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r, c = key
            return self.data[r][c]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            r, c = key
            self.data[r][c] = Fraction(value)
        else:
            raise TypeError("使用 matrix[r, c] = value 格式设置元素")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return self.data == other.data

    def __repr__(self) -> str:
        return f"Matrix({self._format_data()})"

    def __str__(self) -> str:
        return self._format_data()

    def _format_data(self) -> str:
        if not self.data:
            return "[]"
        str_data = []
        for row in self.data:
            str_row = []
            for val in row:
                if val.denominator == 1:
                    str_row.append(str(val.numerator))
                else:
                    str_row.append(str(val))
            str_data.append(str_row)
        col_widths = [max(len(str_data[r][c]) for r in range(self.rows)) for c in range(self.cols)]
        lines = []
        for r in range(self.rows):
            formatted = "  ".join(str_data[r][c].rjust(col_widths[c]) for c in range(self.cols))
            lines.append(f"[ {formatted} ]")
        return "\n".join(lines)

    def to_float_list(self) -> List[List[float]]:
        return [[float(self.data[i][j]) for j in range(self.cols)] for i in range(self.rows)]

    @staticmethod
    def identity(n: int) -> Matrix:
        data = [[0] * n for _ in range(n)]
        for i in range(n):
            data[i][i] = 1
        return Matrix(data)

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        return Matrix([[0] * cols for _ in range(rows)])

    @staticmethod
    def from_input(prompt: str = "") -> Matrix:
        """从用户输入创建矩阵。"""
        if prompt:
            print(prompt)
        while True:
            try:
                row_str = input("输入行数和列数 (如 3 3): ").strip()
                parts = row_str.split()
                rows, cols = int(parts[0]), int(parts[1])
                break
            except (ValueError, IndexError):
                print("格式错误，请重新输入。")
        data = []
        for i in range(rows):
            while True:
                try:
                    row_input = input(f"第 {i+1} 行 (空格分隔): ").strip()
                    row = [Fraction(x) for x in row_input.split()]
                    if len(row) != cols:
                        print(f"需要 {cols} 个元素，请重新输入。")
                        continue
                    data.append(row)
                    break
                except (ValueError, ZeroDivisionError):
                    print("输入无效，请重新输入。")
        return Matrix(data)
