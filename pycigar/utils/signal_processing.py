import numpy as np


###########################################################################
# ##################          BUTTERWORTH           #######################
###########################################################################

# Butterworth polynomial


def butterworth_polynomial(n=1, w0=1.0):
    a0ncoeff = np.zeros([int(n / 2), 3])

    for k1 in range(1, int(n / 2) + 1):
        a0ntemp = np.array([1, -2 * np.cos((2 * k1 + n - 1) / (2 * n) * np.pi), 1])
        a0ncoeff[k1 - 1, :] = a0ntemp

    if n % 2 == 0:
        a0nnorm = np.array([1])
    elif n % 2 == 1:
        a0nnorm = np.array([1, 1])
    a0n = np.zeros(n + 1)

    for k1 in range(0, int(n / 2)):
        a0nnorm = np.convolve(a0nnorm, a0ncoeff[k1, :])
    for k1 in range(0, len(a0nnorm)):
        a0n[k1] = a0nnorm[k1] * w0 ** (n - k1)
    return a0n, a0nnorm


def butterworth_lowpass(n=1, w0=1.0):
    a0n, a0nnorm = butterworth_polynomial(n, w0)
    b0nnorm = np.zeros(n + 1)
    b0nnorm[0] = 1
    b0n = b0nnorm * w0 ** n
    g0n = np.array([b0n, a0n])
    g0nnorm = np.array([b0nnorm, a0nnorm])

    return g0n, g0nnorm


def butterworth_highpass(n=1, w0=1.0):
    a0n, a0nnorm = butterworth_polynomial(n, w0)
    b0nnorm = np.zeros(n + 1)
    b0nnorm[-1] = 1
    b0n = b0nnorm * 1
    g0n = np.array([b0n, a0n])
    g0nnorm = np.array([b0nnorm, a0nnorm])
    return g0n, g0nnorm


def butterworth_bandpass(n=1, w0=1.0):
    if n % 2 == 1:
        pass
    elif n % 2 == 0:
        a0n, a0nnorm = butterworth_polynomial(n, w0)
        b0nnorm = np.zeros(n + 1)
        b0nnorm[int(n / 2)] = np.sqrt(2)
        b0n = b0nnorm * w0 ** (n / 2)
    g0n = np.array([b0n, a0n])
    g0nnorm = np.array([b0nnorm, a0nnorm])
    return g0n, g0nnorm


def butterworth_bandstop(n=1, w0=1.0):
    if n % 2 == 1:
        pass
    elif n % 2 == 0:
        a0n, a0nnorm = butterworth_polynomial(n, w0)
        b0nnorm = np.zeros(n + 1)
        b0nnorm[0] = 1
        b0nnorm[-1] = np.real(-(1j ** n))
        b0n = np.zeros(n + 1)
        b0n[0] = w0 ** n
        b0n[-1] = np.real(-(1j ** n))
    g0n = np.array([b0n, a0n])
    g0nnorm = np.array([b0nnorm, a0nnorm])
    return g0n, g0nnorm


###########################################################################
# ##################              C2D               #######################
###########################################################################
# Continuous to discrete transformation
# g(s) = b(s)/a(s) --> G(z) = B(z)/A(z)

# Continuous time (CT) transfer function g(s) = b(s)/a(s)
# b(s) = b_0 + b_1*s + b_2*s^2 + ... + b_{n-1}*s^(n-1) + b_n*s^n
# a(s) = a_0 + a_1*s + a_2*s^2 + ... + a_{n-1}*s^(n-1) + a_n*s^n
# g(s) is represented by g0n where the first row contains the coefficients of b(s) indexed by ascending powers of s, and the second row contains the coefficients of a(s) indexed by ascending powers of s
# g0n = [[b_0, b_1, ... , b_{n-1}, b_n],[a_0, a_1, ... , a_{n-1}, a_n]]

# Discrete time (DT) transfer function G(z) = B(z)/A(z)
# B(z) = B_0 + B_1*z + B_2*z^2 + ... + B_{n-1}*z^(n-1) + B_n*z^n
# A(z) = A_0 + A_1*z + A_2*z^2 + ... + A_{n-1}*z^(n-1) + A_n*z^n
# G(z) is represented by G0n where the first row contains the coefficients of B(z) indexed by ascending powers of z, and the second row contains the coefficients of A(z) indexed by ascending powers of z
# G0n = [[B_0, B_1, ... , B_{n-1}, B_n],[A_0, A_1, ... , A_{n-1}, A_n]]


# ZOH Transform
# z = exp(sT) ~ 1 + sT
# s ~ (1/T)(z - 1)
def c2dZOH(g0n, T):
    # numerator of transfer function and order
    b0n = g0n[0, :]
    m = np.nonzero(b0n)[0][-1]
    # denominator of transfer function and order
    a0n = g0n[1, :]
    n = np.nonzero(a0n)[0][-1]

    B0n = np.zeros([m + 1, n + 1])
    A0n = np.zeros([n + 1, n + 1])

    # transform for numerator
    # iterate through coefficients of b(s), and therefore powers of s
    for k1 in range(0, m + 1):
        # substitute s = (1/T)(z - 1)
        zz = np.polynomial.polynomial.polypow([-1, 1], k1)
        # multiply by T^n
        zz = zz * T ** (n - k1)
        # multiply by coefficient of b_{k1}
        zz = zz * b0n[k1]
        # separate ascending powers of z and place into array
        B0n[k1, 0 : k1 + 1] = zz

    # transform for denominator
    # iterate through coefficients of a(s), and therefore powers of s
    for k1 in range(0, n + 1):
        # substitute s = (1/T)(z - 1)
        zz = np.polynomial.polynomial.polypow([-1, 1], k1)
        # multiply all terms by T^n
        zz = zz * T ** (n - k1)
        # multiply by coefficient of a_{k1}
        zz = zz * a0n[k1]
        # separate ascending powers of z and place into array
        A0n[k1, 0 : k1 + 1] = zz

    # gather powers of z to obtain coefficients of G(z) = B(z)/A(z)
    B0n = np.sum(B0n, axis=0)
    A0n = np.sum(A0n, axis=0)

    # normalize by A_n
    B0n = B0n / A0n[-1]
    A0n = A0n / A0n[-1]

    G0n = np.array([B0n, A0n])
    return G0n


# Bilinear transform


def c2dbilinear(g0n, T):
    # numerator of transfer function and order
    b0n = g0n[0, :]
    m = np.nonzero(b0n)[0][-1]
    # denominator of transfer function and order
    a0n = g0n[1, :]
    n = np.nonzero(a0n)[0][-1]

    B0n = np.zeros([m + 1, n + 1])
    A0n = np.zeros([n + 1, n + 1])

    # transform for numerator
    # iterate through coefficients of b(s), and therefore powers of s
    for k1 in range(0, m + 1):
        # substitute s ~ (2/T)(z - 1)/(z + 1)
        # multiply all terms by (z+1)^n
        zmtemp = np.polynomial.polynomial.polypow([-1, 1], k1)
        zptemp = np.polynomial.polynomial.polypow([1, 1], n - k1)
        zz = np.convolve(zmtemp, zptemp)
        # multiply all terms by (T/2)^n
        zz = zz * (T / 2) ** (n - k1)
        # mulitply by coefficient b_{k1}
        zz = zz * b0n[k1]
        # separate ascending powers of z and place into array
        B0n[k1, :] = zz

    # transform for denominator
    # iterate through coefficients of a(s), and therefore powers of s
    for k1 in range(0, n + 1):
        # substitute s ~ (2/T)(z - 1)/(z + 1)
        # multiply all terms by (z+1)^n
        zmtemp = np.polynomial.polynomial.polypow([-1, 1], k1)
        zptemp = np.polynomial.polynomial.polypow([1, 1], n - k1)
        zz = np.convolve(zmtemp, zptemp)
        # multiply all terms by (T/2)^n
        zz = zz * (T / 2) ** (n - k1)
        # mulitply by coefficient b_{k1}
        zz = zz * a0n[k1]
        # separate ascending powers of z and place into array
        A0n[k1, :] = zz

    # gather powers of z to obtain coefficients of G(z) = B(z)/A(z)
    B0n = np.sum(B0n, axis=0)
    A0n = np.sum(A0n, axis=0)

    # normalize by A_n
    B0n = B0n / A0n[-1]
    A0n = A0n / A0n[-1]

    G0n = np.array([B0n, A0n])
    return G0n


###########################################################################
# ##################              D2C               #######################
###########################################################################


def d2cZOH(G0n, T):

    B0n = G0n[0, :]
    A0n = G0n[1, :]

    n = np.nonzero(A0n)[0][-1]
    m = np.nonzero(B0n)[0][-1]

    a0n = np.zeros([n + 1, n + 1])
    b0n = np.zeros([m + 1, n + 1])

    for k1 in range(0, n + 1):
        zz = np.polynomial.polynomial.polypow([1, T], k1)
        zz = zz * A0n[k1]
        a0n[k1, 0 : k1 + 1] = zz

    for k1 in range(0, m + 1):
        zz = np.polynomial.polynomial.polypow([1, T], k1)
        zz = zz * B0n[k1]
        b0n[k1, 0 : k1 + 1] = zz

    a0n = np.sum(a0n, axis=0)
    b0n = np.sum(b0n, axis=0)

    b0n = b0n / a0n[-1]
    a0n = a0n / a0n[-1]

    g0n = np.array([b0n, a0n])

    return g0n


# Bilinear transform


def d2cbilinear(G0n, T):

    B0n = G0n[0, :]
    A0n = G0n[1, :]

    n = np.nonzero(A0n)[0][-1]
    m = np.nonzero(B0n)[0][-1]

    a0n = np.zeros([n + 1, n + 1])
    b0n = np.zeros([m + 1, n + 1])

    for k1 in range(0, n + 1):
        sptemp = np.polynomial.polynomial.polypow([2, T], k1)
        smtemp = np.polynomial.polynomial.polypow([2, -T], n - k1)
        ss = np.convolve(smtemp, sptemp)
        ss = ss * A0n[k1]
        a0n[k1, :] = ss

    for k1 in range(0, m + 1):

        sptemp = np.polynomial.polynomial.polypow([2, T], k1)
        smtemp = np.polynomial.polynomial.polypow([2, -T], n - k1)
        ss = np.convolve(smtemp, sptemp)
        ss = ss * B0n[k1]
        b0n[k1, :] = ss

    a0n = np.sum(a0n, axis=0)
    b0n = np.sum(b0n, axis=0)

    b0n = b0n / a0n[-1]
    a0n = a0n / a0n[-1]

    g0n = np.array([b0n, a0n])

    return g0n


###########################################################################
# ##################              FILTERS           #######################
# #########################################################################


def lowpass(n=1, w0=2 * np.pi, zeta=1):

    if n <= 0:
        pass
    elif n == 1:
        b0n = np.array([w0, 0])
        a0n = np.array([w0, 1])
    elif n == 2:
        b0n = np.array([w0 ** 2, 0, 0])
        a0n = np.array([w0 ** 2, 2 * zeta * w0, 1])
    elif n >= 3:
        pass

    g0n = np.array([b0n, a0n])

    return g0n


def highpass(n=1, w0=2 * np.pi, zeta=1):

    if n <= 0:
        pass
    elif n == 1:
        b0n = np.array([0, 1])
        a0n = np.array([w0, 1])
    elif n == 2:
        b0n = np.array([0, 0, 1])
        a0n = np.array([w0 ** 2, 2 * zeta * w0, 1])
    elif n >= 3:
        pass

    g0n = np.array([b0n, a0n])

    return g0n


def bandpass(n=2, w0=2 * np.pi, zeta=1):

    if n <= 1:
        pass
    elif n == 2:
        b0n = np.array([0, 2 * zeta * w0, 0])
        a0n = np.array([w0 ** 2, 2 * zeta * w0, 1])
    elif n >= 3:
        pass

    g0n = np.array([b0n, a0n])

    return g0n


def bandstop(n=2, w0=2 * np.pi, zeta=1):

    if n <= 1:
        pass
    elif n == 2:
        b0n = np.array([w0 ** 2, 0, 1])
        a0n = np.array([w0 ** 2, 2 * zeta * w0, 1])
    elif n >= 3:
        pass

    g0n = np.array([b0n, a0n])

    return g0n
