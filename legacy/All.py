import numpy as np
import os
import time

# 1. 矩阵相加：   shape
# 2. 矩阵相乘：   shape
# 3. 行列式计算   x = y
# 4. 逆矩阵计算   det != 0
# 5. 特征分解     x = y
# 6. 矩阵的转置   None
# 7. 矩阵相减     shape
# 8. 矩阵相除     shape
# 9. 矩阵对应相乘 shape
# 10. 奇异值分解

class MatCal:
    #用来进行两个矩阵进行的操作
    def __init__(self,mat1,mat2):
        self.__mat1 = mat1
        self.__mat2 = mat2

    def add(self):
        if self.__mat1.shape != self.__mat2.shape:
            print("两个待相加矩阵的shape不一致，无法操作")
            return None
        else:
            return self.__mat1 + self.__mat2

    def multi(self):
        #矩阵对应相乘
        if self.__mat1.shape != self.__mat2.shape:
            print("两个待相乘矩阵的shape不满足要求，无法操作")
            return None
        return self.__mat1 * self.__mat2

    def matmulti(self):
        #矩阵相乘
        if self.__mat1.shape[1] != self.__mat2.shape[0]:
            print("两个待相乘矩阵的shape不满足要求，无法操作")
            return None
        return np.matmul(self.__mat1,self.__mat2)

    def sub(self):
        #矩阵相减
        if self.__mat1.shape != self.__mat2.shape:
            print("两个待相减矩阵的shape不满足要求，无法操作")
            return None
        return self.__mat1 - self.__mat2
    def divide(self):
        #矩阵相除
        if self.__mat1.shape != self.__mat2.shape:
            print("两个待相除矩阵的shape不满足要求，无法操作")
            return None
        return self.__mat1 / self.__mat2


class MatCom:
    #用来进行一个矩阵进行操作
    def __init__(self,mat):
        self.__mat = mat

    def det(self):
        #计算行列式
        if self.__mat.shape[0] != self.__mat.shape[1]:
            print("矩阵行列数目应该相等")
            return None
        return np.linalg.det(self.__mat)

    def invert(self):
        #逆矩阵
        det_val = self.det()
        if det_val is not None and abs(det_val) > 1e-10:
            return np.linalg.inv(self.__mat)
        else:
            return None

    def eigs(self):
        #计算矩阵特征分解
        if self.__mat.shape[0] != self.__mat.shape[1]:
            print("矩阵行列数目应该相等")
            return None
        return np.linalg.eig(self.__mat)# 1. 特征值， 2.特征向量

    def svds(self):
        #矩阵奇异值分解
        return np.linalg.svd(self.__mat)# 1.左奇异 2.奇异值 3. 右奇异

    def trans(self):
        #矩阵的转置
        return self.__mat.T

def mkMat(row, col):
    #根据row，col进行矩阵的构造
    print("----start----")
    arr = []
    for i in range(row):
        tmp_arr = []
        r = input("请输入第{:d}行内容:".format(i+1))
        tmp = r.split()
        while len(tmp) != col:
            print("您输入尺寸大小不正确，长度应该为:{:d}".format(col))
            r = input("请输入第{:d}行内容:".format(i+1))
            tmp = r.split()
        for j in range(col):
            try:
                tmp_arr.append(float(tmp[j]))
            except ValueError:
                print("错误：输入的值不是有效的数字，请重新输入")
                return None
        arr.append(tmp_arr)
    print("----end----")
    return np.array(arr,dtype=np.float32)

def printMat(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print("\t{:.1f}".format(mat[i][j]),end="")
        print()

def show2Res(mat1,mat2,mat):
    if mat is None:
        print("错误：矩阵运算失败，无法显示结果")
        return
    print("矩阵:")
    printMat(mat1)
    print("和矩阵:")
    printMat(mat2)
    print("运算结果为:")
    printMat(mat)

if __name__ == "__main__":
    str1 = "请输入想要的操作对应序号：\n 1. 矩阵相加 \n 2. 矩阵相乘 \n 3. 行列式计算 \n 4. 逆矩阵计算 \n 5. 特征分解 \n 6. 矩阵的转置\n "
    str2 = "7. 矩阵相减 \n 8. 矩阵相除 \n 9. 矩阵对应相乘 \n 10. 奇异值分解 \n 0. 退出 \n 请输入："
    while True:
        try:
            a = input(str1+str2)
            if a == "1":
                print("计算两个矩阵相加 （两个矩阵的大小需要一相同）")
                row1,col1 = input("请输入矩阵1的行:"),input("请输入矩阵1的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                except ValueError:
                    print("错误：行和列必须是整数")
                    time.sleep(2)
                    continue
                mat1 = mkMat(row1,col1)
                if mat1 is None:
                    time.sleep(2)
                    continue
                row2,col2 = input("请输入矩阵2的行:"),input("请输入矩阵2的列:")
                try:
                    row2, col2 = int(row2), int(col2)
                except ValueError:
                    print("错误：行和列必须是整数")
                    time.sleep(2)
                    continue
                mat2 = mkMat(row2,col2)
                if mat2 is None:
                    time.sleep(2)
                    continue
                tmp = MatCal(mat1,mat2)
                if tmp.add() is not None:
                    show2Res(mat1,mat2,tmp.add())
            elif a == "2":
                print("计算两个矩阵相乘 （第一个矩阵的第二维与第二个矩阵的第一维需要相同）")
                row1,col1 = input("请输入矩阵1的行:"),input("请输入矩阵1的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat1 = mkMat(row1,col1)
                    if mat1 is None:
                        time.sleep(2)
                        continue
                    row2,col2 = input("请输入矩阵2的行:"),input("请输入矩阵2的列:")
                    row2, col2 = int(row2), int(col2)
                    mat2 = mkMat(row2,col2)
                    if mat2 is None:
                        time.sleep(2)
                        continue
                    tmp = MatCal(mat1,mat2)
                    if tmp.matmulti() is not None:
                        show2Res(mat1,mat2,tmp.matmulti())
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "3":
                print("下面进行行列式计算 (请输入一个矩阵)")
                row1,col1 = input("请输入矩阵的行:"),input("请输入矩阵的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat = mkMat(row1,col1)
                    if mat is None:
                        time.sleep(2)
                        continue
                    tmp = MatCom(mat)
                    if tmp.det() is not None:
                        print("矩阵：")
                        printMat(mat)
                        print("的行列式值为：{:.2f}".format(tmp.det()))
                    else:
                        print("error:请重新输入")
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "4":
                print("进行逆矩阵运算(请输入一个矩阵)")
                row1,col1 = input("请输入矩阵的行:"),input("请输入矩阵的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat = mkMat(row1,col1)
                    if mat is None:
                        time.sleep(2)
                        continue
                    tmp = MatCom(mat)
                    if tmp.invert() is not None:
                        print("矩阵：")
                        printMat(mat)
                        print("的逆矩阵为")
                        printMat(tmp.invert())
                    else:
                        print("error: 请重新输入")
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "5":
                print("进行矩阵特征分解(请输入一个矩阵)")
                row1,col1 = input("请输入矩阵的行:"),input("请输入矩阵的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat = mkMat(row1,col1)
                    if mat is None:
                        time.sleep(2)
                        continue
                    tmp = MatCom(mat)
                    print("矩阵：")
                    printMat(mat)
                    if tmp.eigs() is not None:
                        t1,t2 = tmp.eigs()
                        print("的矩阵特征值为:")
                        print(t1)
                        print("特征向量为:")
                        print(t2)
                    else:
                        print("error:请重新输入")
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "6":
                print("进行矩阵转置运算(请输入一个矩阵)")
                row1,col1 = input("请输入矩阵的行:"),input("请输入矩阵的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat = mkMat(row1,col1)
                    if mat is None:
                        time.sleep(2)
                        continue
                    tmp = MatCom(mat)
                    print("矩阵：")
                    printMat(mat)
                    print("的转置矩阵为")
                    printMat(tmp.trans())
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "7":
                print("计算两个矩阵相减 （两个矩阵的大小需要一相同）")
                row1,col1 = input("请输入矩阵1的行:"),input("请输入矩阵1的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat1 = mkMat(row1,col1)
                    if mat1 is None:
                        time.sleep(2)
                        continue
                    row2,col2 = input("请输入矩阵2的行:"),input("请输入矩阵2的列:")
                    row2, col2 = int(row2), int(col2)
                    mat2 = mkMat(row2,col2)
                    if mat2 is None:
                        time.sleep(2)
                        continue
                    tmp = MatCal(mat1,mat2)
                    if tmp.sub() is not None:
                        show2Res(mat1,mat2,tmp.sub())
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "8":
                print("计算两个矩阵对应相除 （两个矩阵shape应该一样）")
                row1,col1 = input("请输入矩阵1的行:"),input("请输入矩阵1的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat1 = mkMat(row1,col1)
                    if mat1 is None:
                        time.sleep(2)
                        continue
                    row2,col2 = input("请输入矩阵2的行:"),input("请输入矩阵2的列:")
                    row2, col2 = int(row2), int(col2)
                    mat2 = mkMat(row2,col2)
                    if mat2 is None:
                        time.sleep(2)
                        continue
                    tmp = MatCal(mat1,mat2)
                    if tmp.divide() is not None:
                        show2Res(mat1,mat2,tmp.divide())
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "9":
                print("计算两个矩阵对应相乘 （两个矩阵shape应该一样）")
                row1,col1 = input("请输入矩阵1的行:"),input("请输入矩阵1的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat1 = mkMat(row1,col1)
                    if mat1 is None:
                        time.sleep(2)
                        continue
                    row2,col2 = input("请输入矩阵2的行:"),input("请输入矩阵2的列:")
                    row2, col2 = int(row2), int(col2)
                    mat2 = mkMat(row2,col2)
                    if mat2 is None:
                        time.sleep(2)
                        continue
                    tmp = MatCal(mat1,mat2)
                    if tmp.multi() is not None:
                        show2Res(mat1,mat2,tmp.multi())
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "10":
                print("进行矩阵奇异值分解(请输入一个矩阵)")
                row1,col1 = input("请输入矩阵的行:"),input("请输入矩阵的列:")
                try:
                    row1, col1 = int(row1), int(col1)
                    mat = mkMat(row1,col1)
                    if mat is None:
                        time.sleep(2)
                        continue
                    tmp = MatCom(mat)
                    print("矩阵：")
                    printMat(mat)
                    t1,t2,t3 = tmp.svds()
                    print("的左奇异值为:")
                    print(t1)
                    print("奇异值为:")
                    print(t2)
                    print("右奇异值为：")
                    print(t3)
                except ValueError:
                    print("错误：行和列必须是整数")
            elif a == "0":
                break
            else:
                print("输入错误")
            #暂停一段时间
            time.sleep(2)
        except Exception as e:
            print(f"发生错误: {e}")
            time.sleep(2)