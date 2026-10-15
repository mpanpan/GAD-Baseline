import unittest
import numpy as np
from gad_handoff.metrics import evaluate, calibrate_threshold
from gad_handoff.data import NAMES

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0,1,0,1,0,1,0,1])
        self.score=np.array([0.1,.95,.2,.85,.15,.89,.25,.9])
        self.val=np.array([1,1,1,1,0,0,0,0],dtype=bool)
        self.test=~self.val
    def test_perfect_ranking(self):
        r=evaluate(self.y,self.score,self.val,self.test)
        for k in ['AUROC','AUPRC_AP','Recall@K_oracle','F1','Macro-F1','G-Mean']:
            self.assertAlmostEqual(r[k],1)
        self.assertEqual(r['Recall@K_oracle_K'],2)
        self.assertEqual(r['n_test'],4)
    def test_score_polarity(self):
        r=evaluate(self.y,-self.score,self.val,self.test)
        self.assertEqual(r['AUROC'],0)
    def test_refuse_invalid_scores(self):
        bad=self.score.copy();bad[2]=np.nan
        with self.assertRaises(ValueError):evaluate(self.y,bad,self.val,self.test)
    def test_refuse_leakage_split(self):
        with self.assertRaises(ValueError):evaluate(self.y,self.score,self.val,self.val)
    def test_unknown_label_ignored(self):
        yy=self.y.copy();yy[4]=-1
        r=evaluate(yy,self.score,self.val,self.test)
        self.assertEqual(r['n_test'],3)
        self.assertEqual(r['unknown_labels_all_nodes'],1)
    def test_val_threshold_changes_not_with_test_labels(self):
        x=calibrate_threshold(self.y[self.val],self.score[self.val])
        flipped=self.score.copy();flipped[self.test]=flipped[self.test]*100
        y=calibrate_threshold(self.y[self.val],flipped[self.val])
        self.assertEqual(x,y)
    def test_expected_names(self):
        self.assertEqual(NAMES,{'amazon','yelp','elliptic','tfinance'})

if __name__=='__main__':unittest.main()
