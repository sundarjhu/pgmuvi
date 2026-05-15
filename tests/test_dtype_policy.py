"""Dtype-policy conformance tests.

These tests describe the *intended* dtype behaviour documented in
``docs/development/dtype_policy.md``.  Several tests are marked
``@unittest.expectedFailure`` because the current code does not yet
conform to the policy.  Remove the decorator when the corresponding
violation is fixed.
"""

import tempfile
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import (
    make_chromatic_sinusoid_2d,
    make_multi_sinusoid_1d,
    make_multi_sinusoid_chromatic_2d,
    make_simple_sinusoid_1d,
)


# ---------------------------------------------------------------------------
# 1. Direct Lightcurve construction
# ---------------------------------------------------------------------------


class TestLightcurveConstructorDtype(unittest.TestCase):
    """Lightcurve construction should default floating tensors to float64."""

    @unittest.expectedFailure
    def test_numpy_input_xdata_is_float64(self):
        """numpy array inputs should produce float64 xdata.

        Policy: non-tensor numeric inputs must default to torch.float64.
        Currently fails because _ensure_tensor coerces to float32.
        """
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 100, 20))
        y = rng.normal(0, 1, 20)
        yerr = np.full(20, 0.1)
        lc = Lightcurve(t, y, yerr=yerr)
        self.assertEqual(lc.xdata.dtype, torch.float64)

    @unittest.expectedFailure
    def test_numpy_input_ydata_is_float64(self):
        """numpy array inputs should produce float64 ydata.

        Policy: non-tensor numeric inputs must default to torch.float64.
        Currently fails because _ensure_tensor coerces to float32.
        """
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 100, 20))
        y = rng.normal(0, 1, 20)
        yerr = np.full(20, 0.1)
        lc = Lightcurve(t, y, yerr=yerr)
        self.assertEqual(lc.ydata.dtype, torch.float64)

    @unittest.expectedFailure
    def test_numpy_input_yerr_is_float64(self):
        """numpy array inputs should produce float64 yerr.

        Policy: non-tensor numeric inputs must default to torch.float64.
        Currently fails because _ensure_tensor coerces to float32.
        """
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 100, 20))
        y = rng.normal(0, 1, 20)
        yerr = np.full(20, 0.1)
        lc = Lightcurve(t, y, yerr=yerr)
        self.assertEqual(lc.yerr.dtype, torch.float64)

    def test_float64_tensor_input_preserved(self):
        """Explicitly float64 tensor input must stay float64.

        This must already pass: if the input is already a float64 tensor,
        the constructor must not silently down-cast it.
        """
        t = torch.linspace(0, 100, 20, dtype=torch.float64)
        y = torch.randn(20, dtype=torch.float64)
        yerr = torch.full((20,), 0.1, dtype=torch.float64)
        lc = Lightcurve(t, y, yerr=yerr)
        self.assertEqual(lc.xdata.dtype, torch.float64)
        self.assertEqual(lc.ydata.dtype, torch.float64)
        self.assertEqual(lc.yerr.dtype, torch.float64)

    def test_float32_tensor_input_preserved(self):
        """Explicitly float32 tensor input must stay float32.

        Users who explicitly pass float32 should not have their dtype changed.
        This must already pass.
        """
        t = torch.linspace(0, 100, 20, dtype=torch.float32)
        y = torch.randn(20, dtype=torch.float32)
        lc = Lightcurve(t, y)
        self.assertEqual(lc.xdata.dtype, torch.float32)
        self.assertEqual(lc.ydata.dtype, torch.float32)


# ---------------------------------------------------------------------------
# 2. from_csv
# ---------------------------------------------------------------------------


class TestFromCsvDtype(unittest.TestCase):
    """Lightcurve.from_csv should default floating tensors to float64."""

    @unittest.expectedFailure
    def test_from_csv_xdata_is_float64(self):
        """from_csv xdata should default to float64.

        Policy: CSV-loaded numeric data must default to torch.float64.
        Currently fails because _to_float_tensor uses dtype=torch.float32.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("time,flux,flux_err\n")
            for i in range(10):
                fh.write(f"{float(i)},{float(i) * 0.1},{0.05}\n")
            path = fh.name

        lc = Lightcurve.from_csv(path)
        self.assertEqual(lc.xdata.dtype, torch.float64)

    @unittest.expectedFailure
    def test_from_csv_ydata_is_float64(self):
        """from_csv ydata should default to float64.

        Policy: CSV-loaded numeric data must default to torch.float64.
        Currently fails because _to_float_tensor uses dtype=torch.float32.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("time,flux,flux_err\n")
            for i in range(10):
                fh.write(f"{float(i)},{float(i) * 0.1},{0.05}\n")
            path = fh.name

        lc = Lightcurve.from_csv(path)
        self.assertEqual(lc.ydata.dtype, torch.float64)


# ---------------------------------------------------------------------------
# 3. Synthetic light-curve constructors
# ---------------------------------------------------------------------------


class TestSyntheticDtype(unittest.TestCase):
    """Synthetic light-curve generators should return float64 tensors."""

    @unittest.expectedFailure
    def test_make_simple_sinusoid_1d_is_float64(self):
        """make_simple_sinusoid_1d should return a float64 Lightcurve.

        Policy: synthetic constructors must not hard-code float32.
        Currently fails because the function uses dtype=torch.float32.
        """
        lc = make_simple_sinusoid_1d(n_obs=20, seed=0)
        self.assertEqual(lc.xdata.dtype, torch.float64)
        self.assertEqual(lc.ydata.dtype, torch.float64)

    @unittest.expectedFailure
    def test_make_multi_sinusoid_1d_is_float64(self):
        """make_multi_sinusoid_1d should return a float64 Lightcurve.

        Policy: synthetic constructors must not hard-code float32.
        Currently fails because the function uses dtype=torch.float32.
        """
        lc = make_multi_sinusoid_1d(n_obs=20, seed=0)
        self.assertEqual(lc.xdata.dtype, torch.float64)
        self.assertEqual(lc.ydata.dtype, torch.float64)

    @unittest.expectedFailure
    def test_make_chromatic_sinusoid_2d_is_float64(self):
        """make_chromatic_sinusoid_2d should return a float64 Lightcurve.

        Policy: synthetic constructors must not hard-code float32.
        Currently fails because the function uses dtype=torch.float32.
        """
        lc = make_chromatic_sinusoid_2d(n_per_band=10, seed=0)
        self.assertEqual(lc.xdata.dtype, torch.float64)
        self.assertEqual(lc.ydata.dtype, torch.float64)

    @unittest.expectedFailure
    def test_make_multi_sinusoid_chromatic_2d_is_float64(self):
        """make_multi_sinusoid_chromatic_2d should return a float64 Lightcurve.

        Policy: synthetic constructors must not hard-code float32.
        Currently fails because the function uses dtype=torch.float32.
        """
        lc = make_multi_sinusoid_chromatic_2d(n_per_band=10, seed=0)
        self.assertEqual(lc.xdata.dtype, torch.float64)
        self.assertEqual(lc.ydata.dtype, torch.float64)


# ---------------------------------------------------------------------------
# 4. ACF dtype preservation
# ---------------------------------------------------------------------------


class TestACFDtype(unittest.TestCase):
    """ACF output tensors should preserve the owning Lightcurve's dtype."""

    @unittest.expectedFailure
    def test_acf_data_preserves_float64(self):
        """ACF output should be float64 when the Lightcurve is float64.

        Policy: _acf_data must cast back to the owning object's dtype.
        Currently fails because _acf_data calls .float() before returning.
        """
        t = torch.linspace(0, 100, 40, dtype=torch.float64)
        y = torch.sin(2 * torch.pi * t / 20.0)
        yerr = torch.full((40,), 0.1, dtype=torch.float64)
        lc = Lightcurve(t, y, yerr=yerr)

        result = lc.acf(method="data", n_lags=10)
        self.assertEqual(result.lag.dtype, torch.float64)
        self.assertEqual(result.acf.dtype, torch.float64)

    def test_acf_data_preserves_float32(self):
        """ACF output should be float32 when the Lightcurve is float32.

        The current code hard-codes a ``.float()`` cast, which coincidentally
        produces the correct output dtype for float32 input.  This test
        documents and guards that behaviour: it must continue to pass even
        after the implementation is switched to dtype-aware casting.
        """
        t = torch.linspace(0, 100, 40, dtype=torch.float32)
        y = torch.sin(2 * torch.pi * t / 20.0)
        yerr = torch.full((40,), 0.1, dtype=torch.float32)
        lc = Lightcurve(t, y, yerr=yerr)

        result = lc.acf(method="data", n_lags=10)
        self.assertEqual(result.lag.dtype, torch.float32)


# ---------------------------------------------------------------------------
# 5. Integer / boolean tensor dtype invariants
# ---------------------------------------------------------------------------


class TestIntegerBoolDtype(unittest.TestCase):
    """Integer, boolean, and index tensors must keep their correct dtypes.

    These tests document behaviour that should already hold and must continue
    to hold after the dtype refactor.
    """

    def test_isfinite_mask_is_bool(self):
        """torch.isfinite returns a bool tensor."""
        t = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        mask = torch.isfinite(t)
        self.assertEqual(mask.dtype, torch.bool)

    def test_triu_indices_is_long(self):
        """torch.triu_indices returns a long (int64) tensor."""
        idx = torch.triu_indices(5, 5, offset=1)
        self.assertEqual(idx.dtype, torch.long)

    def test_band_array_not_tensor(self):
        """Band labels are stored as a numpy string array, not a tensor.

        This ensures the band attribute is not accidentally coerced to a
        numeric tensor.  Uses a 2-D lightcurve because per-row band labels
        are the natural use case for that path.
        """
        # Build a minimal 2-D (time x wavelength) light curve with two bands.
        n = 10
        t = np.linspace(0, 10, n)
        t2d = np.column_stack([np.tile(t, 2), [550.0] * n + [700.0] * n])
        y = np.ones(2 * n)
        band = np.array(["V"] * n + ["R"] * n)
        lc = Lightcurve(
            torch.as_tensor(t2d, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            band=band,
        )
        self.assertIsNotNone(lc.band)
        # band must not be a float tensor
        self.assertNotIsInstance(lc.band, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
