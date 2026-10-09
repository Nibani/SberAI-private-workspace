"""Numerical compatibility cannot weaken temporal counts, schemas or gates."""
import copy
import math
import unittest
from scripts.verify_temporal_delivery import compare_cells


class TemporalDelivery(unittest.TestCase):
    def sample(self):
        return [{'channel':'absolute','cell':'N4_spike4','n_seeds':64,
                 'representative_failures':1,'fpr_gate':False,'status':'COMPLETE',
                 'persistent_recall':None,'representative_cp_upper95':0.07198975489470942,
                 'false_persistent_object_fraction':0.001953125,
                 'false_persistent_object_fraction_ci95':[0.0,0.005859375]}]

    def test_one_ulp_reports_exact_path_and_passes(self):
        saved=self.sample(); other=copy.deepcopy(saved)
        other[0]['representative_cp_upper95']=math.nextafter(saved[0]['representative_cp_upper95'],math.inf)
        result=compare_cells(saved,other)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(len(result['non_bitexact_floats']),1)
        self.assertEqual(result['non_bitexact_floats'][0]['path'],'cells[0].representative_cp_upper95')
        self.assertGreater(result['max_absolute_difference'],0)

    def test_substantive_float_and_ci_differences_fail(self):
        for key in ['representative_cp_upper95','false_persistent_object_fraction']:
            saved=self.sample(); other=copy.deepcopy(saved); other[0][key]+=1e-7
            self.assertEqual(compare_cells(saved,other)['status'],'FAIL')
        other=self.sample(); other[0]['false_persistent_object_fraction_ci95'][1]+=1e-7
        self.assertEqual(compare_cells(self.sample(),other)['status'],'FAIL')

    def test_counts_booleans_status_and_schema_are_exact(self):
        for key,value in [('n_seeds',63),('representative_failures',0),('fpr_gate',True),
                          ('status','INCOMPLETE'),('channel','relative'),('persistent_recall',0.)]:
            other=self.sample(); other[0][key]=value
            self.assertEqual(compare_cells(self.sample(),other)['status'],'FAIL',key)
        other=self.sample(); other[0]['fpr_gate']=0
        self.assertEqual(compare_cells(self.sample(),other)['status'],'FAIL')
        other=self.sample(); other[0]['n_seeds']=64.0
        self.assertEqual(compare_cells(self.sample(),other)['status'],'FAIL')
        other=self.sample(); del other[0]['status']
        self.assertEqual(compare_cells(self.sample(),other)['status'],'FAIL')
        self.assertEqual(compare_cells(self.sample(),[])['status'],'FAIL')

    def test_nonfinite_floats_fail_even_if_both_nan(self):
        for value in [math.nan,math.inf,-math.inf]:
            saved=self.sample(); other=self.sample()
            saved[0]['representative_cp_upper95']=value; other[0]['representative_cp_upper95']=value
            self.assertEqual(compare_cells(saved,other)['status'],'FAIL')


if __name__=='__main__': unittest.main()
