import clifford as Cf
from pyganja import *
import numpy as np

'''

from clifford.pga import layout, blades

def random_point():
    return blades["e123"] + sum(c * blades[b] for c, b in zip(np.random.randn(3), ["e023", "e013", "e012"]))

gs = GanjaScene()
gs.add_objects([random_point() for i in range(10)], static=False)
with open('test_file.html', 'w') as test_file:
    test_file.write(generate_full_html(str(gs), sig=layout.sig, grid=True, scale=1.0, gl=True))
    draw(gs, sig=layout.sig, grid=True, scale=1.0, gl=True)


'''

from clifford.pga import layout, blades

batch = np.zeros((32, 16))
# convert point :
point_euclidian_dom = np.random.randn(32, 3)
batch[:, 12:15] = point_euclidian_dom
batch[:, 15] = 1

# convert to multivectors
MV_batch = [layout.MultiVector(batch[i, :]) for i in range(batch.shape[0])]


# random color for each vector:
colors = np.random.randint(0, 255, (32, 3))
gs = GanjaScene()
for i in range(batch.shape[0]):
    
    gs.add_object(MV_batch[i], color=colors[i], label=f"point {i}")

with open('points.html', 'w') as test_file:
    test_file.write(generate_full_html(str(gs), sig=layout.sig, grid=True, scale=1.0, gl=True))
    draw(gs, sig=layout.sig, grid=True, scale=1.0, gl=True)