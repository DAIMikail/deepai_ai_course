"""
Convolutional Layers
====================
CNN icin konvolusyon ve pooling katmanlari.

Boyut notasyonu: (batch, channels, height, width) - NCHW format

im2col Optimizasyonu:
    Konvolusyonu matris carpimina donusturur.
    Nested loop yerine tek np.dot kullanarak ~10-50x hiz artisi saglar.

================================================================================
KONVOLUSYON NEDIR?
================================================================================

Konvolusyon, bir filtre (kernel) ile girdi uzerinde kaydirarak nokta carpimi
alma islemidir.

    Girdi (X):           Kernel (W):        Cikti:
    +---+---+---+---+    +---+---+---+
    | 1 | 2 | 3 | 0 |    | 1 | 0 | 1 |     Ilk pencere:
    +---+---+---+---+    +---+---+---+     1*1 + 2*0 + 3*1 +
    | 4 | 5 | 6 | 1 |    | 0 | 1 | 0 |     4*0 + 5*1 + 6*0 +
    +---+---+---+---+    +---+---+---+     7*1 + 8*0 + 9*1 = 29
    | 7 | 8 | 9 | 2 |    | 1 | 0 | 1 |
    +---+---+---+---+    +---+---+---+
    | 3 | 4 | 5 | 6 |
    +---+---+---+---+

================================================================================
MATEMATIKSEL TANIM
================================================================================

Tek kanalli konvolusyon:
    Y[i,j] = sum_{p,q} X[i*s + p, j*s + q] * W[p, q] + b

Cok kanalli (tam) konvolusyon:
    Y[n, f, i, j] = sum_{c,p,q} X[n, c, i*s + p, j*s + q] * W[f, c, p, q] + b[f]

Burada:
    n: ornek indeksi (batch)
    f: filtre indeksi (cikti kanali)
    c: girdi kanali
    i, j: cikti konumu
    p, q: kernel ici konum
    s: stride

================================================================================
CIKTI BOYUTU HESABI
================================================================================

    H_out = (H_in + 2*padding - kH) / stride + 1
    W_out = (W_in + 2*padding - kW) / stride + 1

Ornek: 28x28 girdi, 3x3 kernel, stride=1, padding=0:
    H_out = (28 + 0 - 3) / 1 + 1 = 26

================================================================================
"""

import numpy as np
from .base import Layer


# ============== im2col / col2im Helper Fonksiyonlari ==============

# ==============================================================================
# im2col OPTIMIZASYONU
# ==============================================================================
#
# NEDEN im2col?
# -------------
# Naive konvolusyon 6 nested loop gerektirir (cok yavas):
#
#     # YAVAS VERSIYON - Kullanma!
#     for n in range(m):           # her ornek
#         for f in range(C_out):   # her filtre
#             for i in range(H_out):   # her satir
#                 for j in range(W_out):   # her sutun
#                     for c in range(C_in):    # her kanal
#                         for p in range(kH):      # kernel satir
#                             for q in range(kW):  # kernel sutun
#                                 Y[n,f,i,j] += X[n,c,i*s+p,j*s+q] * W[f,c,p,q]
#
# im2col bu islemi tek matris carpimina donusturur!
#
# im2col MANTIGI
# --------------
# Her konvolusyon penceresini bir sutuna acar:
#
#     Girdi X (1, 1, 4, 4):                3x3 kernel ile pencereler:
#     +---+---+---+---+
#     |x00|x01|x02|x03|                    Pencere 0,0:  Pencere 0,1:
#     +---+---+---+---+                    x00 x01 x02   x01 x02 x03
#     |x10|x11|x12|x13|                    x10 x11 x12   x11 x12 x13
#     +---+---+---+---+                    x20 x21 x22   x21 x22 x23
#     |x20|x21|x22|x23|
#     +---+---+---+---+                    Pencere 1,0:  Pencere 1,1:
#     |x30|x31|x32|x33|                    x10 x11 x12   x11 x12 x13
#     +---+---+---+---+                    x20 x21 x22   x21 x22 x23
#                                          x30 x31 x32   x31 x32 x33
#
#     im2col sonucu (9, 4):   Her sutun bir pencere
#     +-----+-----+-----+-----+
#     | x00 | x01 | x10 | x11 |  <- kernel[0,0] ile carpilacak
#     | x01 | x02 | x11 | x12 |  <- kernel[0,1] ile carpilacak
#     | x02 | x03 | x12 | x13 |  <- kernel[0,2] ile carpilacak
#     | x10 | x11 | x20 | x21 |  <- kernel[1,0] ile carpilacak
#     | x11 | x12 | x21 | x22 |  <- ...
#     | x12 | x13 | x22 | x23 |
#     | x20 | x21 | x30 | x31 |
#     | x21 | x22 | x31 | x32 |
#     | x22 | x23 | x32 | x33 |
#     +-----+-----+-----+-----+
#       ^       ^      ^      ^
#      (0,0)  (0,1)  (1,0)  (1,1) pencere konumlari
#
# ==============================================================================

def im2col(x, kH, kW, stride=1, padding=0):
    """
    Goruntu tensorunu kolon matrisine donustur.

    Her konvolusyon penceresini bir sutuna acar.
    Bu sayede konvolusyon, matris carpimi olarak hesaplanabilir.

    Args:
        x: Girdi tensoru (m, C, H, W)
        kH: Kernel yuksekligi
        kW: Kernel genisligi
        stride: Adim boyutu
        padding: Kenar dolgusu

    Returns:
        col: (C*kH*kW, H_out*W_out*m) matris

    Ornek:
        x: (32, 1, 28, 28) - 32 ornek, 1 kanal, 28x28
        kernel: 3x3, stride=1, padding=0
        col: (1*3*3, 26*26*32) = (9, 21632)
    """
    m, C, H, W = x.shape

    # Padding uygula
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    H_padded, W_padded = x.shape[2], x.shape[3]

    # Cikti boyutlari
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # ===========================================================================
    # STRIDE TRICKS ACIKLAMASI
    # ===========================================================================
    # as_strided bellegi kopyalamadan farkli bir sekilde "goruntuler".
    # Strides, her boyutta bir sonraki elemana ulasmak icin kac byte
    # atlanacagini belirtir.
    #
    # Orijinal x strides: (s0, s1, s2, s3)
    #   s0: sonraki ornege git
    #   s1: sonraki kanala git
    #   s2: sonraki satira git
    #   s3: sonraki sutuna git
    #
    # Yeni strides: (s0, s1, s2, s3, s2*stride, s3*stride)
    #   Son iki boyut: pencere konumlari icin stride atlama
    # ===========================================================================

    # Shape: (m, C, kH, kW, H_out, W_out)
    shape = (m, C, kH, kW, H_out, W_out)

    # Strides hesapla (byte cinsinden atlama miktarlari)
    s = x.strides
    strides = (s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride)

    # as_strided ile pencereler olustur
    cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    # cols: (m, C, kH, kW, H_out, W_out) = (32, 1, 3, 3, 26, 26)

    # (m, C, kH, kW, H_out, W_out) -> (C*kH*kW, H_out*W_out*m)
    # Yeniden sekillendirme: (1, 3, 3, 26, 26, 32) -> (9, 21632)
    cols = cols.transpose(1, 2, 3, 4, 5, 0).reshape(C * kH * kW, -1)

    return cols


# ==============================================================================
# col2im ACIKLAMASI
# ==============================================================================
# im2col'un tersi. Gradyanlari orijinal konumlarina geri yerlestirir.
#
# ONEMLI: Ayni piksel birden fazla pencerede bulunabilir, bu durumda
# gradyanlar TOPLANIR.
#
# NEDEN TOPLAMA?
# --------------
# Bir piksel 3x3 kernel ile stride=1'de 9 farkli pencerede gorunebilir:
#
#     Piksel x[1,1] su pencerelerde:
#       - Pencere (0,0): sag-alt kose
#       - Pencere (0,1): alt-orta
#       - Pencere (1,0): sag-orta
#       - Pencere (1,1): orta
#       - ...
#
#     Her pencereden gelen gradyan katkisi toplanmali.
# ==============================================================================

def col2im(col, x_shape, kH, kW, stride=1, padding=0):
    """
    Kolon matrisini goruntu tensoruna geri donustur (im2col'un tersi).

    Backward pass icin kullanilir.

    Args:
        col: (C*kH*kW, H_out*W_out*m) matris
        x_shape: Orijinal girdi boyutu (m, C, H, W)
        kH: Kernel yuksekligi
        kW: Kernel genisligi
        stride: Adim boyutu
        padding: Kenar dolgusu

    Returns:
        x: Orijinal boyutta tensor (m, C, H, W)
    """
    m, C, H, W = x_shape

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # (C*kH*kW, H_out*W_out*m) -> (C, kH, kW, H_out, W_out, m)
    col_reshaped = col.reshape(C, kH, kW, H_out, W_out, m)

    # (C, kH, kW, H_out, W_out, m) -> (m, C, kH, kW, H_out, W_out)
    col_reshaped = col_reshaped.transpose(5, 0, 1, 2, 3, 4)

    # Padding dahil bos tensor olustur
    x_padded = np.zeros((m, C, H_padded, W_padded))

    # Her pencereyi yerine yerlestir (overlap durumunda topla)
    # += ile overlap'lerde toplama yapiliyor - bu cok onemli!
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += col_reshaped[:, :, :, :, i, j]

    # Padding'i kaldir
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


# ==============================================================================
# CONV2D KATMANI
# ==============================================================================
#
# BOYUTLAR:
#     Input:  (m, C_in, H, W)
#     Kernel: (C_out, C_in, kH, kW)
#     Output: (m, C_out, H_out, W_out)
#
# FORWARD ISLEMININ MATRIS CARPIMI OLARAK GOSTERIMI:
#
#     Y = W_row * X_col + b
#
#     W_row (16, 9):          x_col (9, 21632):           out (16, 21632):
#     +-------------+         +---------------+           +---------------+
#     | f0: 9 agirlik|    @    | 21632 pencere |     =     | f0: 21632 cikti|
#     | f1: 9 agirlik|         | her biri 9 elem|          | f1: 21632 cikti|
#     | ...         |         |               |           | ...           |
#     | f15:9 agirlik|         |               |           | f15:21632 cikti|
#     +-------------+         +---------------+           +---------------+
#
# ==============================================================================

class Conv2D(Layer):
    """
    2D Konvolusyon Katmani.

    Args:
        in_channels: Girdi kanal sayisi (orn: grayscale=1, RGB=3)
        out_channels: Cikti kanal sayisi (filtre sayisi)
        kernel_size: Filtre boyutu (int veya tuple)
        stride: Adim boyutu (default=1)
        padding: Kenar dolgusu (default=0)
        seed: Random seed

    Boyutlar:
        Input:  (m, C_in, H, W)
        Output: (m, C_out, H_out, W_out)

        H_out = (H - kernel_size + 2*padding) // stride + 1
        W_out = (W - kernel_size + 2*padding) // stride + 1

    Ornek:
        conv = Conv2D(1, 32, kernel_size=3)
        out = conv.forward(x)  # x: (m, 1, 28, 28) -> out: (m, 32, 26, 26)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # He initialization - ReLU ile kullanildiginda iyi calisir
        if seed is not None:
            np.random.seed(seed)

        kH, kW = self.kernel_size
        fan_in = in_channels * kH * kW

        # Kernel: (C_out, C_in, kH, kW)
        self.W = np.random.randn(out_channels, in_channels, kH, kW) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((out_channels, 1))

        # Gradyanlar
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Konvolusyon islemi (im2col optimizasyonlu).

        Args:
            x: Girdi (m, C_in, H, W)

        Returns:
            out: Cikti (m, C_out, H_out, W_out)

        Islem Adim Adim:
            1. im2col ile girdiyi matrise donustur
            2. W'yi satirlara ac
            3. Matris carpimi yap
            4. Sonucu tekrar 4D tensore donustur
        """
        m, C_in, H, W = x.shape  # (32, 1, 28, 28)
        kH, kW = self.kernel_size  # (3, 3)

        # Cikti boyutlari
        H_out = (H + 2 * self.padding - kH) // self.stride + 1  # 26
        W_out = (W + 2 * self.padding - kW) // self.stride + 1  # 26

        # ===========================================================================
        # ADIM 1: im2col
        # Girdiyi (C_in*kH*kW, H_out*W_out*m) matrise donustur
        # (1*3*3, 26*26*32) = (9, 21632)
        # ===========================================================================
        x_col = im2col(x, kH, kW, self.stride, self.padding)

        # ===========================================================================
        # ADIM 2: W'yi duzlestir
        # (C_out, C_in, kH, kW) -> (C_out, C_in*kH*kW)
        # (16, 1, 3, 3) -> (16, 9)
        # ===========================================================================
        W_row = self.W.reshape(self.out_channels, -1)

        # ===========================================================================
        # ADIM 3: Matris carpimi
        # (C_out, C_in*kH*kW) @ (C_in*kH*kW, H_out*W_out*m) + (C_out, 1)
        # (16, 9) @ (9, 21632) + (16, 1) = (16, 21632)
        # ===========================================================================
        out = np.dot(W_row, x_col) + self.b

        # ===========================================================================
        # ADIM 4: Yeniden sekillendirme
        # (C_out, H_out*W_out*m) -> (C_out, H_out, W_out, m) -> (m, C_out, H_out, W_out)
        # (16, 21632) -> (16, 26, 26, 32) -> (32, 16, 26, 26)
        # ===========================================================================
        out = out.reshape(self.out_channels, H_out, W_out, m).transpose(3, 0, 1, 2)

        # Cache - backward icin sakla
        self.cache['x'] = x
        self.cache['x_col'] = x_col

        return out

    def backward(self, dout):
        """
        Geri yayilim (im2col optimizasyonlu).

        Args:
            dout: Sonraki katmandan gelen gradyan (m, C_out, H_out, W_out)

        Returns:
            dx: Onceki katmana iletilecek gradyan (m, C_in, H, W)

        =========================================================================
        TUREV HESAPLAMALARI
        =========================================================================

        Notasyon:
            dL/dY = dout (bir sonraki katmandan gelen gradyan)
            dL/dW = dW (agirlik gradyani)
            dL/db = db (bias gradyani)
            dL/dX = dx (bir onceki katmana iletilecek gradyan)

        Forward'da: Y = W_row * X_col + b

        -------------------------------------------------------------------------
        dW Turevi (Chain Rule):
        -------------------------------------------------------------------------
            dL/dW_row = dL/dY * dY/dW_row = dL/dY * X_col^T

            Kod olarak:
            # dout: (m, C_out, H_out, W_out) = (32, 16, 26, 26)
            # Reshape: (C_out, H_out*W_out*m) = (16, 21632)
            # x_col: (9, 21632)
            # dW = dout @ x_col.T
            # (16, 21632) @ (21632, 9) = (16, 9)

        -------------------------------------------------------------------------
        db Turevi:
        -------------------------------------------------------------------------
            dL/db = sum_{n,i,j} dL/dY[n,:,i,j]

            Her filtre icin tum konumlar ve ornekler uzerinden toplam.

            Kod olarak:
            # dout: (m, C_out, H_out, W_out)
            # axis=(0,2,3): ornek, yukseklik, genislik uzerinden topla
            # db: (16, 1)

        -------------------------------------------------------------------------
        dx Turevi (Chain Rule):
        -------------------------------------------------------------------------
            dL/dX_col = W_row^T * dL/dY

            Kod olarak:
            # W_row: (16, 9)
            # dout_reshaped: (16, 21632)
            # dx_col = W.T @ dout
            # (9, 16) @ (16, 21632) = (9, 21632)

            Sonra col2im ile orijinal sekle donustur.
        =========================================================================
        """
        x = self.cache['x']
        x_col = self.cache['x_col']

        m, C_in, H, W = x.shape
        _, C_out, H_out, W_out = dout.shape
        kH, kW = self.kernel_size

        # ===========================================================================
        # ADIM 1: dout'u matris formuna getir
        # (m, C_out, H_out, W_out) -> (C_out, H_out*W_out*m)
        # (32, 16, 26, 26) -> (16, 21632)
        # ===========================================================================
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(C_out, -1)

        # W'yi reshape: (C_out, C_in*kH*kW) = (16, 9)
        W_row = self.W.reshape(self.out_channels, -1)

        # ===========================================================================
        # ADIM 2: dW hesapla
        # dW = dout @ x_col.T
        # (C_out, H_out*W_out*m) @ (H_out*W_out*m, C_in*kH*kW)
        # (16, 21632) @ (21632, 9) = (16, 9)
        # ===========================================================================
        dW_row = np.dot(dout_reshaped, x_col.T)
        self.dW = dW_row.reshape(self.W.shape) / m  # Orijinal sekle dondur ve ortala

        # ===========================================================================
        # ADIM 3: db hesapla
        # db = sum(dout) over (batch, height, width)
        # axis=(0, 2, 3): ornek, yukseklik, genislik uzerinden topla
        # ===========================================================================
        self.db = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1) / m

        # ===========================================================================
        # ADIM 4: dx_col hesapla
        # dx_col = W.T @ dout
        # (C_in*kH*kW, C_out) @ (C_out, H_out*W_out*m)
        # (9, 16) @ (16, 21632) = (9, 21632)
        # ===========================================================================
        dx_col = np.dot(W_row.T, dout_reshaped)

        # ===========================================================================
        # ADIM 5: col2im ile dx'e donustur
        # (9, 21632) -> (32, 1, 28, 28)
        # ===========================================================================
        dx = col2im(dx_col, x.shape, kH, kW, self.stride, self.padding)

        return dx

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def get_grads(self):
        return {'dW': self.dW, 'db': self.db}

    def __repr__(self):
        return f"Conv2D({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# ==============================================================================
# MAXPOOL2D KATMANI
# ==============================================================================
#
# Her pencerede maksimum degeri secer:
#     Y[n, c, i, j] = max_{p,q} X[n, c, i*s + p, j*s + q]
#
# GORSEL:
#     Girdi (1 kanal gosterimi):     2x2 MaxPool:
#     +----+----+----+----+          +----+----+
#     | 1  | 3  | 2  | 1  |          | 6  | 8  |  max(1,3,5,6)=6, max(2,1,4,8)=8
#     +----+----+----+----+    ->    +----+----+
#     | 5  | 6  | 4  | 8  |          | 9  | 7  |  max(2,4,9,3)=9, max(1,2,5,7)=7
#     +----+----+----+----+          +----+----+
#     | 2  | 4  | 1  | 2  |
#     +----+----+----+----+
#     | 9  | 3  | 5  | 7  |
#     +----+----+----+----+
#
# ==============================================================================

class MaxPool2D(Layer):
    """
    2D Max Pooling Katmani.

    Args:
        pool_size: Pooling pencere boyutu (default=2)
        stride: Adim boyutu (default=pool_size)

    Boyutlar:
        Input:  (m, C, H, W)
        Output: (m, C, H//pool_size, W//pool_size)

    Ornek:
        pool = MaxPool2D(pool_size=2)
        out = pool.forward(x)  # x: (m, 32, 26, 26) -> out: (m, 32, 13, 13)
    """

    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, x):
        """
        Max pooling islemi.

        Her pencerede maksimum degeri secer.

        Args:
            x: Girdi (m, C, H, W)

        Returns:
            out: Cikti (m, C, H_out, W_out)
        """
        m, C, H, W = x.shape  # (32, 16, 26, 26)
        pH = pW = self.pool_size  # 2

        H_out = (H - pH) // self.stride + 1  # 13
        W_out = (W - pW) // self.stride + 1  # 13

        out = np.zeros((m, C, H_out, W_out))  # (32, 16, 13, 13)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + pH
                w_start = j * self.stride
                w_end = w_start + pW

                # 2x2 pencere al: (32, 16, 2, 2)
                x_slice = x[:, :, h_start:h_end, w_start:w_end]

                # Her pencerede max bul
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))

        self.cache['x'] = x

        return out

    def backward(self, dout):
        """
        Geri yayilim - max degerin konumuna gradyan ilet.

        =========================================================================
        MAXPOOL BACKWARD ACIKLAMASI
        =========================================================================

        MaxPool turevsiz bir fonksiyon (argmax), ama subgradient kullaniriz:

            dY/dX[p,q] = 1  eger X[p,q] = max(pencere)
                         0  aksi halde

        Gradyan sadece maksimum degerin konumuna iletilir.

        GORSEL:
            Forward:                         Backward:
            +----+----+                      +----+----+
            | 1  | 3  |  -> max=6 -> dout=0.5| 0  | 0  |
            +----+----+                      +----+----+
            | 5  | 6* |  (* = max konum)     | 0  |0.5 | <- gradyan sadece max'a
            +----+----+                      +----+----+
        =========================================================================

        Args:
            dout: Sonraki katmandan gelen gradyan (m, C, H_out, W_out)

        Returns:
            dx: Onceki katmana iletilecek gradyan (m, C, H, W)
        """
        x = self.cache['x']
        m, C, H, W = x.shape  # (32, 16, 26, 26)
        pH = pW = self.pool_size  # 2
        _, _, H_out, W_out = dout.shape  # (32, 16, 13, 13)

        dx = np.zeros_like(x)  # (32, 16, 26, 26)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + pH
                w_start = j * self.stride
                w_end = w_start + pW

                # 2x2 pencere al: (32, 16, 2, 2)
                x_slice = x[:, :, h_start:h_end, w_start:w_end]

                # Max degerini bul: (32, 16, 1, 1)
                max_vals = np.max(x_slice, axis=(2, 3), keepdims=True)

                # Max konumlarinin maskesi (boolean): (32, 16, 2, 2)
                # True sadece max degerin oldugu yerde
                mask = (x_slice == max_vals)

                # Gradyani sadece max konumuna ilet
                # dout[:,:,i,j] shape: (32, 16)
                # [:,:,None,None] ile (32, 16, 1, 1) yapiyoruz -> broadcast
                dx[:, :, h_start:h_end, w_start:w_end] += mask * dout[:, :, i, j][:, :, None, None]

        return dx

    def __repr__(self):
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride})"


# ==============================================================================
# FLATTEN KATMANI
# ==============================================================================
#
# CNN ciktisini Dense katmana uygun formata donusturur.
#
# BOYUT DONUSUMU:
#     Input:  (m, C, H, W)  - CNN format
#     Output: (C*H*W, m)    - Dense format
#
# NOT: Dense katmanlar (features, samples) formati bekliyor,
#      bu yuzden transpose yapiliyor.
#
# ==============================================================================

class Flatten(Layer):
    """
    Flatten Katmani - cok boyutlu tensoru 1D'ye duzlestir.

    Conv/Pool ciktilarini Dense katmana baglamak icin kullanilir.

    Boyutlar:
        Input:  (m, C, H, W)
        Output: (C*H*W, m)  <- Dense katman formatina uygun (features, samples)

    Ornek:
        flatten = Flatten()
        out = flatten.forward(x)  # x: (m, 32, 13, 13) -> out: (5408, m)
    """

    def forward(self, x):
        """
        Flatten islemi.

        Args:
            x: Girdi (m, C, H, W) = (32, 16, 13, 13)

        Returns:
            out: Cikti (C*H*W, m) = (2704, 32)
        """
        self.cache['input_shape'] = x.shape
        m = x.shape[0]

        # (m, C, H, W) -> (m, C*H*W) -> (C*H*W, m)
        # (32, 16, 13, 13) -> (32, 2704) -> (2704, 32)
        out = x.reshape(m, -1).T

        return out

    def backward(self, dout):
        """
        Geri yayilim - orijinal sekle geri dondur.

        Flatten'da ogrenilabilir parametre yok, sadece sekil degisikligi.

        Args:
            dout: Sonraki katmandan gelen gradyan (C*H*W, m) = (2704, 32)

        Returns:
            dx: Onceki katmana iletilecek gradyan (m, C, H, W) = (32, 16, 13, 13)
        """
        input_shape = self.cache['input_shape']

        # (C*H*W, m) -> (m, C*H*W) -> (m, C, H, W)
        # (2704, 32) -> (32, 2704) -> (32, 16, 13, 13)
        dx = dout.T.reshape(input_shape)

        return dx

    def __repr__(self):
        return "Flatten()"


# ==============================================================================
# TAM FORWARD-BACKWARD AKISI
# ==============================================================================
#
# FORWARD (->):
#
# Input         Conv2D        ReLU      MaxPool    Flatten       Dense        ReLU        Dense      Softmax
# (m,1,28,28) -> (m,16,26,26) -> (ayni) -> (m,16,13,13) -> (2704,m) -> (64,m) -> (ayni) -> (10,m) -> (10,m)
#      |            |                        |             |          |               |         |
#    cache       cache                    cache        cache      cache           cache     cache
#                W,b                                               W,b             W,b
#
# ================================================================================
#                                   Loss = CrossEntropy(Y_pred, Y)
# ================================================================================
#
# BACKWARD (<-):
#
#   dx           dx           dx          dx           dx          dx           dx        dL/dy
# (m,1,28,28) <- (m,16,26,26) <- (ayni) <- (m,16,13,13) <- (2704,m) <- (64,m) <- (ayni) <- (10,m) <- (10,m)
#                |                        |                         |               |
#               dW,db                    (yok)                    dW,db           dW,db
#
# ==============================================================================
#
# OZET TABLO:
# -----------
# | Katman   | Forward                  | Backward dx              | Ogrenilen Parametreler   |
# |----------|--------------------------|--------------------------|--------------------------|
# | Conv2D   | Y = W * X_col + b        | dX_col = W^T * dY        | dW = dY * X_col^T, db    |
# | ReLU     | Y = max(0, X)            | dX = dY * (X > 0)        | Yok                      |
# | MaxPool  | Y = max(pencere)         | dX[max_idx] = dY         | Yok                      |
# | Flatten  | Reshape                  | Reshape geri             | Yok                      |
# | Dense    | Y = WX + b               | dX = W^T * dY            | dW = dY * X^T, db        |
# | Softmax  | Y_i = e^X_i / sum(e^X_j) | (CE ile birlikte)        | Yok                      |
#
# ==============================================================================
