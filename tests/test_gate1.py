import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None

@unittest.skipUnless(torch is not None, 'PyTorch optional for Gate 0-only environments')
class GateOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from gate1.models import PhaseGraph, GRUMemory, FastWeightMemory, LearnedWindowAttention
        cls.models = [PhaseGraph, GRUMemory, FastWeightMemory, LearnedWindowAttention]

    def test_split_scenario_target_uses_latest_write(self):
        from gate1.data import episodes
        b = episodes(64,20,torch.Generator().manual_seed(13),scenario='rewrite')
        self.assertTrue(torch.equal(b.keys[:,-1],b.query))
        self.assertTrue(torch.equal(b.bits[:,-1],b.target))
        for i in range(b.keys.size(0)):
            j = torch.where(b.keys[i,:-1]==b.query[i])[0][-1]
            self.assertNotEqual(int(b.bits[i,j]),int(b.target[i]))

    def test_path_pair_differs_except_endpoints(self):
        from gate1.models import build_paths
        a,b = build_paths()
        self.assertEqual(a.shape,(63,64))
        self.assertTrue(torch.allclose(a.sum(-1),torch.full((63,),12.)))
        self.assertTrue(torch.allclose(b.sum(-1),torch.full((63,),12.)))
        difference=a-b
        for i in range(63):
            nonzeros = torch.where(difference[i]!=0)[0].tolist()
            self.assertEqual(sorted(nonzeros),[0,i+1])

    def test_model_streaming_gradients_shapes_and_bytes(self):
        from gate1.data import episodes
        from gate1.models import q8
        sample = episodes(5,12,torch.Generator().manual_seed(12))
        for cls in self.models:
            model = cls()
            outputs, memory = model(sample.keys,sample.bits,sample.query)
            for name,logit in outputs.items():
                self.assertEqual(logit.shape,(5,), (cls.__name__,name))
                self.assertTrue(torch.isfinite(logit).all().item())
            loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(logit, sample.target.float()) for logit in outputs.values())
            loss.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))
            if isinstance(memory,tuple):
                self.assertEqual(memory[0].shape[1], min(sample.keys.shape[1],32))
            else:
                self.assertEqual(memory[0].numel(),64)

    def test_quantized_round_trip_and_writer_dependence(self):
        from gate1.models import q8, PhaseGraph
        t = torch.linspace(0,1,100)
        q = q8(t)
        self.assertTrue(torch.allclose(q, ((q*255).round().to(torch.uint8).float()/255)))
        graph=PhaseGraph()
        keys=torch.tensor([[11,20,19],[11,20,19]])
        bits=torch.tensor([[0,1,0],[0,1,1]])
        m=graph.write_state(keys,bits)
        self.assertGreater((m[0]-m[1]).abs().sum().item(),0.)
        out=graph.waves(m,torch.tensor([11,11]))
        self.assertEqual(out[0].shape,(2,))

    def test_exact_baselines_and_counterfactual(self):
        from gate1.models import exact_window,exact_indexed
        keys=torch.tensor([[1,5,7,5,11]])
        bits=torch.tensor([[0,1,0,0,1]])
        query=torch.tensor([5])
        found,x=exact_window(keys,bits,query)
        self.assertTrue(found.item())
        self.assertEqual(x.item(),0)
        self.assertEqual(exact_indexed(keys,bits,query).item(),0)

if __name__ == '__main__':
    unittest.main()
