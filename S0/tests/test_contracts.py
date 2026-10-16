import json
import tempfile
import unittest
from pathlib import Path
from vgec.schemas import adapt,model_view
from vgec.evaluation.diagnostics import evaluate,classification,analyze_pool
from vgec.export.submission import export_submission
from vgec.data.split import inner_split
from vgec.inference.verifier import candidate,select_s0


class Contracts(unittest.TestCase):
    def row(self,label=1,refs=None):
        return adapt({"ID":25,"Input":"Câu gốc.","Output":{"Error_Type_Id":label,"Fixed Sentence":refs if refs is not None else ["Câu sửa.","Câu hợp lệ."]}},"dev")

    def test_multiple_references_and_unicode_raw(self):
        r=self.row()
        self.assertEqual(len(r["references_raw"]),2)
        text="a\u0301"
        self.assertEqual(model_view(text),"á")
        self.assertEqual(adapt({"ID":"s","Input":text},"test",False)["source_raw"],text)

    def test_inference_never_needs_output(self):
        r=adapt({"ID":1,"Input":"Câu chưa có nhãn."},"test",False)
        self.assertIsNone(r["label"])
        self.assertEqual(r["references_raw"],[])

    def test_bad_label_and_missing_error_references(self):
        with self.assertRaises(ValueError):
            self.row(label=5)
        with self.assertRaises(ValueError):
            self.row(refs=[])

    def test_no_error_empty_refs_explicit_identity(self):
        r=self.row(label=0,refs=[])
        self.assertEqual(r["references_raw"],[])
        self.assertEqual(r["correction_references"],[r["source_raw"]])

    def test_incomplete_targets_excluded_but_raw_retained(self):
        r=self.row(refs=["Câu...", "Câu sửa."])
        self.assertEqual(r["references_raw"],["Câu...","Câu sửa."])
        self.assertEqual(r["correction_references"],["Câu sửa."])

    def test_multireference_em_and_fixed_label_set(self):
        r=self.row()
        p=dict(uid=r["uid"],id=r["id"],source_raw=r["source_raw"],predicted_label=1,selected_text="Câu hợp lệ.")
        metrics=evaluate([p],[r])
        self.assertEqual(metrics["correction"]["em_raw"],1.)
        self.assertEqual(metrics["classification"]["macro_f1"],.2)
        self.assertIsNone(metrics["GLEU+"])

    def test_prediction_integrity(self):
        r=self.row()
        with self.assertRaises(ValueError):
            evaluate([], [r])
        p=dict(uid=r["uid"],id=r["id"],source_raw="other",predicted_label=1,selected_text="Câu sửa.")
        with self.assertRaises(ValueError):
            evaluate([p],[r])

    def test_micro_f1_equals_accuracy(self):
        m=classification([0,1,2,3,4],[0,1,1,3,0])
        self.assertEqual(m["micro_f1"],.6)

    def test_truncation_fallback(self):
        copy=candidate("raw","raw","copy")
        bad=candidate("raw","partial","bart",truncated=True)
        self.assertEqual(select_s0([bad,copy])["text_raw"],"raw")

    def test_submission_order_and_gate(self):
        inputs=[{"ID":"b","Input":"b"},{"ID":1,"Input":"a"}]
        pred=[dict(id=1,source_raw="a",predicted_label=1,selected_text="fixed"),dict(id="b",source_raw="b",predicted_label=0,selected_text="b")]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"submission.json"
            with self.assertRaises(ValueError):
                export_submission(pred,inputs,path)
            export_submission(pred,inputs,path,draft=True)
            result=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([r["ID"] for r in result],["b",1])
            self.assertEqual(result[0]["Output"]["Fixed Sentence"],["b"])

    def test_inner_group_split_is_disjoint(self):
        rows=[dict(uid=f"r{i}",group_id=f"g{i//2}",label=(i//2)%5) for i in range(60)]
        a,b=inner_split(rows)
        self.assertFalse({r["group_id"] for r in a}&{r["group_id"] for r in b})
        self.assertEqual({r["label"] for r in b},set(range(5)))


if __name__=="__main__":
    unittest.main()
