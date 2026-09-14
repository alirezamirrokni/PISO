"""Two orthogonal sketches must retain the already learned coordinate."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import yaml
from src.ablations.algorithm import run_piso_variant

class LinearProblem:
    n=2
    initial_x=np.zeros(2)
    full=np.array([2.,3.])
    known=np.array([.3,.4])
    def sample_demands(self,x,count,rng):return np.tile(self.known,(count,1))
    def partial_gradients(self,observations):return observations
    def sample_losses(self,x,count,rng):return np.full(count,x@self.full),None
    def evaluate(self,x,count,rng):return float(x@self.full)

class AblationTest(unittest.TestCase):
    def test_centered_recursion_and_readout(self):
        config=yaml.safe_load((Path(__file__).parents[1]/'configs/ablations.yaml').read_text())
        for template in ('piso','piso2'):
            with self.subTest(template=template):
                p={**config['defaults'][template], 'gamma':1.3,'rho':.8,
                   'step_size':.01,'step_decay':1.,'radius_decay':1.,'batch_increment':0}
                states=[];directions=[np.sqrt(2)*np.eye(2)[i] for i in range(2)]
                def draw(*args):
                    u=directions[len(states)]
                    return [u],[u]
                with patch('src.ablations.algorithm.sample_directions',side_effect=draw):
                    trace=run_piso_variant(LinearProblem(),p,base_seed=1,week_id='08',
                        run_index=0,variant_seed_key='test',max_samples=180,metric_samples=1,
                        checkpoint_every=1,load_checkpoint=lambda:None,
                        save_checkpoint=lambda v:states.append(deepcopy(v['state'])),
                        clear_checkpoint=lambda:None)
                v=LinearProblem.full-p['gamma']*LinearProblem.known
                expected=.4*v  # each coordinate measured once with ||u||^2=2
                np.testing.assert_allclose(states[-1]['momentum'],expected,atol=1e-12)
                b=.2+.8*(p['lambda'] if template=='piso2' else 0.)
                expected_x=-.01*(2*p['gamma']*LinearProblem.known+
                                  np.array([(2*b+.4)*v[0],2*b*v[1]]))
                np.testing.assert_allclose(trace.final_x,expected_x,atol=1e-12)
                self.assertFalse(trace.failed)

if __name__=='__main__':unittest.main()
