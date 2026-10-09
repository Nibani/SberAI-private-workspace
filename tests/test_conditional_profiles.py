import csv
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sbercluster import conditional_profiles as cp
from sbercluster.panel import read_panel, CATEGORIES, PERIODS
from scripts.run_conditional_profiles import load_inputs, dump
from scripts.verify_conditional_profiles import verify


class ConditionalContracts(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(73)
        level=rng.normal(10,1,40)
        shares=rng.uniform(1,20,(40,5))
        status=np.array(['городской округ']*20+['муниципальный район']*20)
        labor=np.column_stack([rng.normal(-2,.3,40),rng.uniform(.2,.8,40)])
        return level,shares,status,labor

    def test_heldout_outcome_immunity(self):
        level,shares,status,labor=self.fixture()
        a=cp.fit_transform(level[:30],shares[:30],status[:30],labor[:30])
        # Public API takes neither wage nor legacy labels. Changing heldout fields
        # cannot alter a training-only conditional transform or its prototypes.
        level[30:]+=100; shares[30:]*=20; labor[30:]*=10
        b=cp.fit_transform(level[:30],shares[:30],status[:30],labor[:30])
        self.assertTrue(np.array_equal(a.center,b.center))
        self.assertTrue(np.array_equal(a.ridge.coef_,b.ridge.coef_))
        self.assertFalse(any('wage' in p for p in inspect.signature(cp.fit_transform).parameters))

    def test_constant_projection(self):
        x=np.array([[0.,0.],[2.,0.],[0.,2.],[2.,2.]])
        centers=x.copy()
        q=cp.nearest(x,centers)
        # Deliberately disagreeing graph labels are irrelevant to q(reference).
        graph=3-q
        self.assertTrue(np.any(graph!=q))
        repeated=np.repeat(x[:,None],24,axis=1)
        projected=cp.nearest(repeated.reshape(-1,2),centers).reshape(4,24)
        self.assertEqual(int(np.sum(projected!=q[:,None])),0)

    def test_zero_iqr_and_unknown_type(self):
        level,shares,status,labor=self.fixture()
        labor[:,1]=.5
        tf=cp.fit_transform(level,shares,status,labor)
        self.assertFalse(tf.active[6])
        c=tf.coordinates(level,shares,status,labor)
        self.assertTrue(np.isfinite(cp.block_attributes(c,tf.active)).all())
        self.assertTrue(np.all(c[:,6]==0))
        unknown=cp.status_design(['unseen'],tf.statuses)
        self.assertEqual(unknown[0,-1],1)
        self.assertEqual(unknown[0,:-1].sum(),0)

    def test_wage_filter_only_outcome(self):
        rows={f'tid_{i}':{'entity_id':f'tid_{i}','region_name':'r','municipal_district_type':'district',
                         'population_total':'1000','wage_total':str(0 if i==0 else 100),
                         cp.LABOR_FIELDS[0]:'.2',cp.LABOR_FIELDS[1]:'.5'} for i in range(4)}
        rows['tid_2'][cp.LABOR_FIELDS[0]]='';rows['tid_3'][cp.LABOR_FIELDS[1]]=''
        panel=SimpleNamespace(ids=np.array(list(rows)),regions=np.array(['r']*4))
        with tempfile.TemporaryDirectory() as directory:
            bridge=Path(directory)/'bridge.csv'
            bridge.write_text('entity_id\n'+'\n'.join(rows)+'\n',encoding='utf-8')
            config={'inputs':{'panel':'unused','cohort2023':'unused23','cohort2024':'unused24','bridge':str(bridge)},
                    'cohort2024_sha256':'testhash'}
            with patch('scripts.run_conditional_profiles.sha',return_value='testhash'),patch('scripts.run_conditional_profiles.read_panel',return_value=panel),patch('scripts.run_conditional_profiles.economic',return_value=rows):
                result=load_inputs(config)
        self.assertEqual(result['labor_valid'].tolist(),[True,True,False,False])
        self.assertEqual(result['outcome'].tolist(),[False,True,False,False])

    def test_graph_ties_and_order(self):
        ids=np.array(['b','a','c','d'])
        unit=np.ones((4,3))/np.sqrt(3)
        first=cp.directed_temporal_graph(unit,ids)
        permutation=np.array([3,1,0,2])
        second=cp.directed_temporal_graph(unit[permutation],ids[permutation])
        mapped=lambda e,i:[(str(i[a]),str(i[b])) for a,b in e]
        self.assertEqual(set(mapped(first,ids)),set(mapped(second,ids[permutation])))
        self.assertEqual([ids[b] for a,b in first if a==0],['a','c','d'])
        self.assertFalse(any(a==b for a,b in first))

    def test_panel_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[]
            for tid in [2,1]:
                for period in reversed(PERIODS):
                    rows.append({'entity_id':f'tid_{tid}','territory_id':str(tid),'name':str(tid),
                                 'region':'r','period':period,'total_rub':100+tid,
                                 **{f'{c}_pct':2+tid for c in CATEGORIES}})
            paths=[Path(directory)/f'{i}.csv' for i in range(2)]
            for path,data in zip(paths,[rows,list(reversed(rows))]):
                with path.open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(data)
            a,b=map(read_panel,paths)
            self.assertTrue(np.array_equal(a.ids,b.ids))
            self.assertTrue(np.array_equal(a.totals,b.totals))

    def test_folds_are_region_atomic_and_order_independent(self):
        regions=['r1','r2','r3','r1','r4','r5','r6']
        self.assertEqual(cp.region_folds(regions),cp.region_folds(list(reversed(regions))))
        self.assertEqual(len(cp.region_folds(regions)),6)

    def test_regional_two_way_centering(self):
        level,shares,status,labor=self.fixture()
        tf=cp.fit_transform(level,shares,status,labor)
        lv=np.repeat(level[:,None],12,axis=1)
        sp=np.repeat(shares[:,None],12,axis=1)
        regions=np.array(['A']*20+['B']*20)
        _,rms,diag=cp.temporal_vectors(tf,lv,sp,status,regions)
        self.assertTrue(np.all(rms<1e-12))
        self.assertEqual(diag['zero_norm_nodes'],40)

    def test_external_arithmetic_equal_region(self):
        rows=[]
        for method in cp.METHODS:
            for i,(region,target) in enumerate([('A',1.),('A',1.),('B',3.)]):
                rows.append({'entity_id':str(i),'region':region,'method':method,'target':target,'prediction':0.})
        result=cp.external_arithmetic(rows)
        self.assertAlmostEqual(result['losses']['C']['region_mse'],5.)
        self.assertAlmostEqual(result['losses']['C']['municipality_mse'],11/3)
        self.assertFalse(result['type_external_increment'])

    def test_indicator_columns_are_not_continuous(self):
        control_width=5
        self.assertEqual(cp.continuous_columns('C4',control_width),[0,1])
        self.assertEqual(cp.continuous_columns('PFULL4',control_width),[0,1,*range(5,12),12])
        self.assertEqual(cp.continuous_columns('PQUADG',control_width),[*range(54),57])

    def test_incomplete_cannot_pass_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'summary.json').write_text('{"status":"INCOMPLETE"}',encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'Incomplete'):
                verify(output)

    def test_numpy_scalar_export_is_native_without_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'test.json'
            dump(target,{'valid':np.bool_(True),'variance':np.float64(.5)})
            self.assertEqual(json.loads(target.read_text()),{'valid':True,'variance':.5})
            with self.assertRaises(ValueError):
                dump(target,{'variance':np.float64(np.inf)})


if __name__=='__main__':
    unittest.main()
