"""矩阵计算器 CLI 交互界面"""

import sys
from fractions import Fraction
from .matrix import Matrix
from . import operations as ops


HELP_TEXT = """
╔══════════════════════════════════════════════════════════╗
║              矩阵计算器 v1.0 — 命令帮助                  ║
╠══════════════════════════════════════════════════════════╣
║  new          创建新矩阵（手动输入）                      ║
║  id <n>       创建 n×n 单位矩阵                          ║
║  zero <r> <c> 创建 r×c 零矩阵                            ║
║  add          矩阵加法  (A + B)                           ║
║  sub          矩阵减法  (A - B)                           ║
║  mul          矩阵乘法  (A × B)                           ║
║  smul         标量乘法  (k · A)                           ║
║  trans        矩阵转置  (A^T)                             ║
║  det          行列式    (det A)                           ║
║  inv          逆矩阵    (A^(-1))                          ║
║  rank         矩阵的秩  (rank A)                          ║
║  ref          行阶梯形  (REF)                             ║
║  rref         简化行阶梯形 (RREF)                         ║
║  solve        解线性方程组 Ax = b                         ║
║  lu           LU 分解    (A = LU)                         ║
║  eig2         2×2 矩阵特征值                              ║
║  cofactor     余子式矩阵                                  ║
║  adjugate     伴随矩阵                                    ║
║  show         显示当前矩阵                                ║
║  swap <a> <b> 交换矩阵 A 和 B                            ║
║  help         显示本帮助                                  ║
║  quit / exit  退出程序                                    ║
╚══════════════════════════════════════════════════════════╝

当前矩阵存储位置: A (第一个矩阵), B (第二个矩阵)
"""


def parse_matrix_literal() -> Matrix:
    """解析用户输入的矩阵字面量。"""
    print("  输入矩阵（每行用空格分隔数字，空行结束）:")
    print("  示例: 1 2 3")
    print("        4 5 6")
    print("        (空行结束)")
    data = []
    while True:
        line = input("  > ").strip()
        if not line:
            if data:
                break
            print("  矩阵不能为空，请重新输入:")
            continue
        try:
            row = [Fraction(x) for x in line.split()]
            data.append(row)
        except (ValueError, ZeroDivisionError) as e:
            print(f"  输入错误: {e}，请重新输入该行:")
            data.pop() if data else None
            continue
    return Matrix(data)


def get_two_matrices():
    """获取两个矩阵（A 和 B）。"""
    print("\n--- 矩阵 A ---")
    a = parse_matrix_literal()
    print("\n--- 矩阵 B ---")
    b = parse_matrix_literal()
    return a, b


def cli():
    """CLI 主循环。"""
    print("=" * 58)
    print("       矩阵计算器 v1.0 (Matrix Calculator)")
    print("       CS101 课程大作业 — Python 3")
    print("=" * 58)
    print("输入 help 查看命令列表，输入 quit 退出。\n")

    matrix_a = None
    matrix_b = None

    while True:
        try:
            cmd = input("\n>>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not cmd:
            continue

        if cmd in ("quit", "exit", "q"):
            print("再见！")
            break

        if cmd == "help":
            print(HELP_TEXT)
            continue

        if cmd == "new":
            try:
                m = parse_matrix_literal()
                if matrix_a is None:
                    matrix_a = m
                    print("已存储到 A:")
                else:
                    matrix_b = m
                    print("已存储到 B:")
                print(m)
            except Exception as e:
                print(f"错误: {e}")
            continue

        if cmd.startswith("id"):
            try:
                n = int(cmd.split()[1])
                m = Matrix.identity(n)
                if matrix_a is None:
                    matrix_a = m
                else:
                    matrix_b = m
                print(m)
            except (ValueError, IndexError):
                print("用法: id <n>")
            continue

        if cmd.startswith("zero"):
            try:
                parts = cmd.split()
                r, c = int(parts[1]), int(parts[2])
                m = Matrix.zeros(r, c)
                if matrix_a is None:
                    matrix_a = m
                else:
                    matrix_b = m
                print(m)
            except (ValueError, IndexError):
                print("用法: zero <rows> <cols>")
            continue

        if cmd == "show":
            if matrix_a:
                print("A =")
                print(matrix_a)
            if matrix_b:
                print("B =")
                print(matrix_b)
            if not matrix_a and not matrix_b:
                print("当前没有矩阵，请先用 new 创建。")
            continue

        if cmd.startswith("swap"):
            try:
                parts = cmd.split()
                x, y = parts[1].upper(), parts[2].upper()
                if x == "A" and y == "B":
                    matrix_a, matrix_b = matrix_b, matrix_a
                    print("已交换 A 和 B。")
                elif x == "B" and y == "A":
                    matrix_a, matrix_b = matrix_b, matrix_a
                    print("已交换 A 和 B。")
                else:
                    print("用法: swap a b")
            except IndexError:
                print("用法: swap a b")
            continue

        # 一元运算
        unary_ops = {
            "trans": (transpose, "转置"),
            "det": (lambda m: print(f"det = {ops.determinant(m)}"), "行列式"),
            "inv": (ops.inverse, "逆矩阵"),
            "rank": (lambda m: print(f"rank = {ops.rank(m)}"), "秩"),
            "ref": (lambda m: ops.row_echelon(m)[0], "行阶梯形"),
            "rref": (ops.reduced_row_echelon, "简化行阶梯形"),
            "cofactor": (ops.cofactor_matrix, "余子式矩阵"),
            "adjugate": (ops.adjugate, "伴随矩阵"),
        }

        if cmd in unary_ops:
            m = matrix_a or matrix_b
            if m is None:
                print("请先创建矩阵。")
                continue
            try:
                result = unary_ops[cmd][0](m)
                if isinstance(result, Matrix):
                    print(f"{unary_ops[cmd][1]} =")
                    print(result)
                    matrix_b = result
                    print("(结果已存储到 B)")
            except Exception as e:
                print(f"错误: {e}")
            continue

        # 二元运算
        if cmd in ("add", "sub", "mul"):
            a, b = get_two_matrices()
            try:
                if cmd == "add":
                    result = ops.matrix_add(a, b)
                    print("A + B =")
                elif cmd == "sub":
                    result = ops.matrix_sub(a, b)
                    print("A - B =")
                else:
                    result = ops.matrix_mul(a, b)
                    print("A × B =")
                print(result)
                matrix_a = a
                matrix_b = b
            except Exception as e:
                print(f"错误: {e}")
            continue

        if cmd == "smul":
            try:
                k = Fraction(input("  标量 k = ").strip())
            except (ValueError, ZeroDivisionError):
                print("  无效的标量")
                continue
            m = matrix_a or matrix_b
            if m is None:
                print("请先创建矩阵。")
                continue
            result = ops.scalar_mul(k, m)
            print(f"{k} · A =")
            print(result)
            matrix_b = result
            continue

        if cmd == "solve":
            try:
                print("\n--- 系数矩阵 A ---")
                a = parse_matrix_literal()
                print("\n--- 常数列向量 b ---")
                b = parse_matrix_literal()
                x = ops.solve(a, b)
                print("\n解 x =")
                print(x)
            except Exception as e:
                print(f"错误: {e}")
            continue

        if cmd == "lu":
            m = matrix_a or matrix_b
            if m is None:
                print("请先创建矩阵。")
                continue
            try:
                L, U = ops.lu_decompose(m)
                print("L =")
                print(L)
                print("\nU =")
                print(U)
                # 验证
                product = ops.matrix_mul(L, U)
                print("\nL × U =")
                print(product)
            except Exception as e:
                print(f"错误: {e}")
            continue

        if cmd == "eig2":
            m = matrix_a or matrix_b
            if m is None:
                print("请先创建矩阵。")
                continue
            try:
                lam1, lam2 = ops.eigenvalues_2x2(m)
                print(f"λ₁ = {lam1}")
                print(f"λ₂ = {lam2}")
            except Exception as e:
                print(f"错误: {e}")
            continue

        print(f"未知命令: {cmd}，输入 help 查看帮助。")


# 需要导入 transpose 用于一元运算映射
from .operations import transpose
