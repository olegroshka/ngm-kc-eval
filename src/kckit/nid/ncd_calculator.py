import zlib
import lzma
import bz2

from src.kckit.nid.nid_interface import InformationDistanceCalculator

class NCDCalculator(InformationDistanceCalculator):
    def __init__(self, compression_method='zlib'):
        """
        Initialize the NCD class with a specified compression method.
        Supports 'zlib', 'lzma', and 'bz2'. Defaults to 'zlib' if not specified.
        """
        self.methods = {
            'zlib': self._zlib_compression,
            'lzma': self._lzma_compression,
            'bz2': self._bz2_compression
        }

        if compression_method not in self.methods:
            raise ValueError(
                f"Unsupported compression method: {compression_method}. Choose from 'zlib', 'lzma', or 'bz2'.")

        self.compress = self.methods[compression_method]

    def _zlib_compression(self, data):
        """Default compression method using zlib."""
        return len(zlib.compress(data))

    def _lzma_compression(self, data):
        """Compression using lzma."""
        return len(lzma.compress(data))

    def _bz2_compression(self, data):
        """Compression using bz2."""
        return len(bz2.compress(data))

    def compute_ncd(self, x, y):
        """
        Compute the Normalized Compression Distance (NCD) between two strings x and y.
        """
        x_bytes = x.encode() if isinstance(x, str) else x
        y_bytes = y.encode() if isinstance(y, str) else y

        cx, cy, cxy = self.compress(x_bytes), self.compress(y_bytes), self.compress(x_bytes + y_bytes)

        return (cxy - min(cx, cy)) / max(cx, cy)

    def compute_distance(self, data1, data2):
        return self.compute_ncd(data1, data2)

    def train_dictionary(self, samples):
        # NCD doesn't need dictionary training. So, we just pass.
        pass


if __name__ == "__main__":
    # Example Usage:
    ncd_calculator_zlib = NCDCalculator('zlib')
    ncd_calculator_lzma = NCDCalculator('lzma')
    ncd_calculator_bz2 = NCDCalculator('bz2')

    x = "The cat sat on the mat."
    y = "The dog barked at the cat."

    print("Using zlib:", ncd_calculator_zlib.compute_ncd(x, y))
    print("Using lzma:", ncd_calculator_lzma.compute_ncd(x, y))
    print("Using bz2:", ncd_calculator_bz2.compute_ncd(x, y))
