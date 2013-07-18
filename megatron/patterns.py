
from sympy.matrices.expressions import MatrixSymbol, Transpose, MatMul
from sympy import Symbol, ask, Q, Dummy, ImmutableMatrix, BlockMatrix

old = locals().copy()

# Pattern variables
alpha = Symbol('_alpha')
beta  = Symbol('_beta')
n,m,k = map(Symbol, ['_n', '_m', '_k'])
i1, i2, i3, i4, i5, i6 = map(Symbol, ['_i%d'%i for i in range(1, 7)])
A = MatrixSymbol('_A', n, m)
B = MatrixSymbol('_B', m, k)
C = MatrixSymbol('_C', n, k)
D = MatrixSymbol('_D', n, n)
X = MatrixSymbol('_X', n, m)
Y = MatrixSymbol('_Y', n, m)
Z = MatrixSymbol('_Z', n, n)
S = MatrixSymbol('_S', n, n)
x = MatrixSymbol('_x', n, 1)
a = MatrixSymbol('_a', m, 1)
b = MatrixSymbol('_b', k, 1)
SW, SX, SY, SZ = [MatrixSymbol('_S'+s, n, n) for s in 'WXYZ']
blocks = ImmutableMatrix([[SW, SX], [SY, SZ]])

new = locals().copy()

vars = [v for (k, v) in new.items() if k not in old and k != 'old']

from computations.matrices.blas import GEMM, SYMM, AXPY, SYRK
from computations.matrices.lapack import GESV, POSV, IPIV, LASWP, GESVLASWP
from computations.matrices.fftw import FFTW, IFFTW
from computations.matrices.blocks import Slice, Join
from computations.matrices.elemental import ElemProd
from computations.matrices.permutation import PermutationMatrix
from sympy.matrices.expressions import ZeroMatrix, HadamardProduct
from sympy.matrices.expressions.fourier import DFT

comp_to_comp = [
    (GEMM(alpha, A, B, beta, C), SYMM(alpha, A, B, beta, C), Q.symmetric(A) | Q.symmetric(B)),
    ]


# pattern is (source expression, target expression, wilds, condition)
blas = [
    (A*A.T, SYRK(1.0, A, 0.0, ZeroMatrix(A.rows, A.rows)), True),
    (A.T*A, SYRK(1.0, A.T, 0.0, ZeroMatrix(A.cols, A.cols)), True),
    (alpha*A*B + beta*C, SYMM(alpha, A, B, beta, C), SYMM.condition),
    (alpha*A*B + C, SYMM(alpha, A, B, 1.0, C), SYMM.condition),
    (A*B + beta*C, SYMM(1.0, A, B, beta, C), SYMM.condition),
    (A*B + C, SYMM(1.0, A, B, 1.0, C), SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), SYMM.condition),
    (A*B, SYMM(1.0, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), SYMM.condition),

    (alpha*A*B + beta*C, GEMM(alpha, A, B, beta, C), True),
    (alpha*A*B + C, GEMM(alpha, A, B, 1.0, C), True),
    (A*B + beta*C, GEMM(1.0, A, B, beta, C), True),
    (A*B + C, GEMM(1.0, A, B, 1.0, C), True),
    (alpha*A*B, GEMM(alpha, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), True),
    (A*B, GEMM(1.0, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), True),

    (alpha*X + Y, AXPY(alpha, X, Y), AXPY.condition),
    (X + Y, AXPY(1.0, X, Y), True)
]

lapack = [
    (Z.I*X, POSV(Z, X), Q.symmetric(Z) & Q.positive_definite(Z)),
    (Z.I*X, GESVLASWP(Z, X), True),
    # (Z.I*X, GESV(Z, X) + LASWP(MatMul(PermutationMatrix(IPIV(Z.I*X)), Z.I*X), IPIV(Z.I*X)), True),

]


other = [
    (DFT(n)*x, FFTW(x), True),
    (DFT(n).T*x, IFFTW(x), True),
    (HadamardProduct(A, X), ElemProd(A, X), True),
    (BlockMatrix(blocks), Join(BlockMatrix(blocks)), True),
    (X[i1:i2:i3, i4:i5:i6], Slice(X[i1:i2:i3, i4:i5:i6]), True),
]

patterns = lapack + blas + other
