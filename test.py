from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
student = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],['intel', 'diff'], [2, 2])
student.add_cpds(cpd_d, cpd_i, cpd_g)
inference = BayesianModelSampling(student)
print inference.forward_sample(size=3, return_type='recarray')