import unittest, numpy as np, pandas as pd
from run import ewma,garch_fit,garch_forecast
class Tests(unittest.TestCase):
 def test_ewma_uses_past(self):
  x=pd.Series([1.,2.,3.]); self.assertEqual(ewma(x)[1],.94+.06)
 def test_garch_positive(self):
  x=np.random.default_rng(1).normal(0,.01,200); p=garch_fit(pd.Series(x)); self.assertLess(p[1]+p[2],1); self.assertTrue((garch_forecast(x,p)>0).all())
if __name__=='__main__': unittest.main()
