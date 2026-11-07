import unittest

import numpy as np

import g2l_nn as g


class DirectRedshiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.theory, cls.space, _ = g.build()
        cls.theta = cls.space.fiducial()
        cls.cosmo, cls.hod = cls.space.split(cls.theta)
        cls.bg = cls.theory.background(cls.cosmo)

    def test_nn_receives_exact_grid_and_hod_for_every_lens(self):
        theory = self.theory
        original = theory.emu.ln_pk
        seen = []

        def capture(features):
            seen.append(features.copy())
            return original(features)

        theory.emu.ln_pk = capture
        try:
            spectra = theory.nn_grid(self.cosmo, self.hod)
        finally:
            theory.emu.ln_pk = original

        expected = []
        for i, lens in enumerate(theory.lens_names):
            idx = np.flatnonzero(theory.nl_z[lens] != 0)
            self.assertEqual(len(spectra[lens]), len(idx))
            for z in theory.z_l[idx]:
                expected.append(np.r_[self.cosmo, self.hod[i], z])
        np.testing.assert_array_equal(seen[0], np.asarray(expected))

    def test_direct_matrix_matches_individual_nn_evaluations(self):
        theory = self.theory
        spectra = theory.nn_grid(self.cosmo, self.hod)
        for i, lens in enumerate(theory.lens_names):
            actual = theory.pk_matrix_z(lens, spectra[lens], self.bg)
            for index in theory.z_indices[lens]:
                features = np.r_[self.cosmo, self.hod[i], theory.z_l[index]]
                ln_pk = theory.emu.ln_pk(features)[0, 2]
                expected = np.exp(g._interp_extrap(
                    np.log(theory.ell_c / self.bg["chi_l"][index]), theory.emu.lnk, ln_pk))
                np.testing.assert_allclose(actual[index], expected, rtol=1e-12)
            np.testing.assert_array_equal(actual[theory.nl_z[lens] == 0], 0)
        prediction = theory.model_vector(self.theta, self.space)
        self.assertEqual(prediction.shape, (self.data.n,))
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_redshift_outside_training_range_is_rejected(self):
        class LimitedEmulator:
            has_z = True
            z_range = (0.0, 0.01)

        with self.assertRaisesRegex(ValueError, "ausserhalb des NN-Trainingsbereichs"):
            g.G2LTheory(self.data.combos, self.data.theta, self.theory.zs,
                        self.theory.n_lenses, self.theory.n_sources, emulator=LimitedEmulator())

    def test_legacy_emulator_still_predicts(self):
        emulator = g.GMEmulator(g.EMU_FILE_Z0)
        self.assertFalse(emulator.has_z)
        theory = g.G2LTheory(self.data.combos, self.data.theta, self.theory.zs,
                            self.theory.n_lenses, self.theory.n_sources, emulator=emulator)
        prediction = theory.model_vector(self.theta, self.space)
        self.assertEqual(prediction.shape, (self.data.n,))
        self.assertTrue(np.all(np.isfinite(prediction)))


if __name__ == "__main__":
    unittest.main()
