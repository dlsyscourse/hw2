import sys
sys.path.append("./python")
import numpy as np
import needle as ndl
import needle.nn as nn

import mugrade

sys.path.append("./apps")
from mlp_resnet import *
"""Deterministically generate a matrix"""
def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(low, high, size=shape))

def check_prng(*shape):
    """ We want to ensure that numpy generates random matrices on your machine/colab
        Such that our tests will make sense
        So this matrix should match our to full precision
    """
    return get_tensor(*shape).cached_data

def batchnorm_forward(*shape, affine=False):
    x = get_tensor(*shape)
    bn = ndl.nn.BatchNorm(shape[1])
    if affine:
        bn.weight.data = get_tensor(shape[1], entropy=42)
        bn.bias.data = get_tensor(shape[1], entropy=1337)
    return bn(x).cached_data

def batchnorm_backward(*shape, affine=False):
    x = get_tensor(*shape)
    bn = ndl.nn.BatchNorm(shape[1])
    if affine:
        bn.weight.data = get_tensor(shape[1], entropy=42)
        bn.bias.data = get_tensor(shape[1], entropy=1337)
    y = (bn(x)**2).sum().backward()
    return x.grad.cached_data

def batchnorm_running_mean(*shape, iters=10):
    bn = ndl.nn.BatchNorm(shape[1])
    for i in range(iters):
        x = get_tensor(*shape, entropy=i)
        y = bn(x)
    return bn.running_mean.cached_data

def batchnorm_running_var(*shape, iters=10):
    bn = ndl.nn.BatchNorm(shape[1])
    for i in range(iters):
        x = get_tensor(*shape, entropy=i)
        y = bn(x)
    return bn.running_var.cached_data

def batchnorm_running_grad(*shape, iters=10):
    bn = ndl.nn.BatchNorm(shape[1])
    for i in range(iters):
        x = get_tensor(*shape, entropy=i)
        y = bn(x)
    bn.eval()
    (y**2).sum().backward()
    return x.grad.cached_data

def relu_forward(*shape):
    f = ndl.nn.ReLU()
    x = get_tensor(*shape)
    return f(x).cached_data

def relu_backward(*shape):
    f = ndl.nn.ReLU()
    x = get_tensor(*shape)
    (f(x)**2).sum().backward()
    return x.grad.cached_data

def layernorm_forward(shape, dims):
    f = ndl.nn.LayerNorm(dims)
    x = get_tensor(*shape)
    return f(x).cached_data

def layernorm_backward(shape, dims):
    f = ndl.nn.LayerNorm(dims)
    x = get_tensor(*shape)
    (f(x)**4).sum().backward()
    return x.grad.cached_data

def softmax_loss_forward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = ndl.nn.SoftmaxLoss()
    return np.array(f(x, y).cached_data)

def softmax_loss_backward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = ndl.nn.SoftmaxLoss()
    loss = f(x, y)
    loss.backward()
    return x.grad.cached_data

def linear_forward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = ndl.nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    return f(x).cached_data

def linear_backward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = ndl.nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    (f(x)**2).sum().backward()
    return x.grad.cached_data

def sequential_forward(batches=3):
    np.random.seed(42)
    f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
    x = get_tensor(batches, 5)
    return f(x).cached_data

def sequential_backward(batches=3):
    np.random.seed(42)
    f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
    x = get_tensor(batches, 5)
    f(x).sum().backward()
    return x.grad.cached_data

def residual_forward(shape=(5,5)):
    np.random.seed(42)
    f = nn.Residual(nn.Sequential(nn.Linear(*shape), nn.ReLU(), nn.Linear(*shape[::-1])))
    x = get_tensor(*shape[::-1])
    return f(x).cached_data

def residual_backward(shape=(5,5)):
    np.random.seed(42)
    f = nn.Residual(nn.Sequential(nn.Linear(*shape), nn.ReLU(), nn.Linear(*shape[::-1])))
    x = get_tensor(*shape[::-1])
    f(x).sum().backward()
    return x.grad.cached_data

def learn_model_1d(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = get_tensor(1024, feature_size).cached_data
    y = get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    for _ in range(epochs):
        for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
            opt.reset_grad()
            X0, y0 = ndl.Tensor(X0, dtype="float64"), ndl.Tensor(y0)
            out = model(X0)
            loss = loss_func(out, y0)
            loss.backward()
            # Opt should not change gradients.
            grad_before = model.parameters()[0].grad.detach().cached_data
            opt.step()
            grad_after = model.parameters()[0].grad.detach().cached_data
            np.testing.assert_allclose(grad_before, grad_after, rtol=1e-5, atol=1e-5, \
                                       err_msg="Optim should not modify gradients in place")


    return np.array(loss.cached_data)

def learn_model_1d_eval(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = get_tensor(1024, feature_size).cached_data
    y = get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
        opt.reset_grad()
        X0, y0 = ndl.Tensor(X0, dtype="float64"), ndl.Tensor(y0)
        out = model(X0)
        loss = loss_func(out, y0)
        loss.backward()
        opt.step()

    X_test = ndl.Tensor(get_tensor(batch, feature_size).cached_data)
    y_test = ndl.Tensor(get_int_tensor(batch, low=0, high=nclasses).cached_data.astype(np.uint8))

    model.eval()

    return np.array(loss_func(model(X_test), y_test).cached_data)

def init_a_tensor_of_shape(shape, init_fn):
    x = get_tensor(*shape)
    np.random.seed(42)
    init_fn(x)
    return x.cached_data

def global_tensor_count():
    return np.array(ndl.autograd.TENSOR_COUNTER)

def nn_linear_weight_init():
    np.random.seed(1337)
    f = ndl.nn.Linear(7, 4)
    return f.weight.cached_data

def nn_linear_bias_init():
    np.random.seed(1337)
    f = ndl.nn.Linear(7, 4)
    return f.bias.cached_data

class UselessModule(ndl.nn.Module):
    def __init__(self):
        super().__init__()
        self.stuff = {'layer1': nn.Linear(4, 4),
                      'layer2': [nn.Dropout(0.1), nn.Sequential(nn.Linear(4, 4))]}

    def forward(self, x):
        raise NotImplementedError()

def check_training_mode():
    model = nn.Sequential(
        nn.BatchNorm(4),
        nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 4),
            nn.Dropout(0.1),
        ),
        nn.Linear(4, 4),
        UselessModule()
    )

    model_refs = [
        model.modules[0],
        model.modules[1].modules[0],
        model.modules[1].modules[1],
        model.modules[1].modules[2],
        model.modules[2],
        model.modules[3],
        model.modules[3].stuff['layer1'],
        model.modules[3].stuff['layer2'][0],
        model.modules[3].stuff['layer2'][1].modules[0]
    ]

    eval_mode = [1 if not x.training else 0 for x in model_refs]
    model.eval()
    eval_mode.extend([1 if not x.training else 0 for x in model_refs])
    model.train()
    eval_mode.extend([1 if not x.training else 0 for x in model_refs])

    return np.array(eval_mode)

def power_scalar_forward(shape, power=2):
    x = get_tensor(*shape)
    return (x**power).cached_data

def power_scalar_backward(shape, power=2):
    x = get_tensor(*shape)
    y = (x**power).sum()
    y.backward()
    return x.grad.cached_data

def logsoftmax_forward(shape, mult=1.0):
    x = get_tensor(*shape) * mult
    return ndl.ops.logsoftmax(x).cached_data

def logsoftmax_backward(shape, mult=1.0):
    x = get_tensor(*shape)
    y = ndl.ops.logsoftmax(x * mult)
    z = (y**2).sum()
    z.backward()
    return x.grad.cached_data

def dropout_forward(shape, prob=0.5):
    np.random.seed(3)
    x = get_tensor(*shape)
    f = nn.Dropout(prob)
    return f(x).cached_data

def dropout_backward(shape, prob=0.5):
    np.random.seed(3)
    x = get_tensor(*shape)
    f = nn.Dropout(prob)
    y = f(x).sum()
    y.backward()
    return x.grad.cached_data

def residual_block_num_params(dim, hidden_dim, norm):
    model = ResidualBlock(dim, hidden_dim, norm)
    return np.array(num_params(model))

def residual_block_forward(dim, hidden_dim, norm, drop_prob):
    np.random.seed(2)
    input_tensor = ndl.Tensor(np.random.randn(1, dim))
    output_tensor = ResidualBlock(dim, hidden_dim, norm, drop_prob)(input_tensor)
    return output_tensor.numpy()

def mlp_resnet_num_params(dim, hidden_dim, num_blocks, num_classes, norm):
    model = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm)
    return np.array(num_params(model))

def mlp_resnet_forward(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
    np.random.seed(4)
    input_tensor = ndl.Tensor(np.random.randn(2, dim))
    output_tensor = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)(input_tensor)
    return output_tensor.numpy()

def train_epoch_1(hidden_dim, batch_size, optimizer, **kwargs):
    np.random.seed(1)
    model = MLPResNet(784, hidden_dim)

    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")

    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             collate_fn=ndl.data.collate_mnist)

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    model.eval()

    return np.array(train_epoch(train_dataloader, model, nn.SoftmaxLoss(), opt))


def eval_epoch_1(hidden_dim, batch_size):
    np.random.seed(1)
    model = MLPResNet(784, hidden_dim)

    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")

    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False,
             collate_fn=ndl.data.collate_mnist,
             drop_last=False)

    loss_func = nn.SoftmaxLoss()

    model.train()

    return np.array(evaluate(test_dataloader, model, nn.SoftmaxLoss()))

def train_mnist_1(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim):
    np.random.seed(1)
    out = train_mnist(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim, data_dir="./data")
    return np.array(out)


def test_check_prng_contact_us_if_this_fails_1():
	np.testing.assert_allclose(check_prng(3, 3),
		np.array([[2.1 , 0.95, 3.45],
		 [3.1 , 2.45, 2.3 ],
		 [3.3 , 0.4 , 1.2 ]], dtype=np.float32), rtol=1e-08, atol=1e-08)

def test_op_power_scalar_forward_1():
	np.testing.assert_allclose(power_scalar_forward((2,2), power=2),
		np.array([[11.222499, 17.639997],
		 [ 0.0625 , 20.25 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_power_scalar_forward_2():
	np.testing.assert_allclose(power_scalar_forward((2,2), power=-1.5),
		np.array([[0.16309206, 0.11617859],
		 [8. , 0.10475656]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_power_scalar_backward_1():
	np.testing.assert_allclose(power_scalar_backward((2,2), power=2),
		np.array([[6.7, 8.4],
		 [0.5, 9. ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_op_power_scalar():
	mugrade.submit(power_scalar_forward((3,1), power=1.3))
	mugrade.submit(power_scalar_forward((1,3), power=-0.3))
	mugrade.submit(power_scalar_backward((3,3), power=-0.4))


def test_op_logsoftmax_forward_1():
	np.testing.assert_allclose(logsoftmax_forward((3, 3)),
		np.array([[-1.6436583 , -2.7936583 , -0.29365814],
		 [-0.6787312 , -1.3287311 , -1.4787312 ],
		 [-0.16337626, -3.0633762 , -2.2633762 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsoftmax_stable_forward_1():
	np.testing.assert_allclose(logsoftmax_forward((3, 3), mult=1e5),
		np.array([[-135000.02, -250000. , 0. ],
		 [ 0. , -65000. , -80000. ],
		 [ 0. , -290000. , -210000. ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsoftmax_backward_1():
	np.testing.assert_allclose(logsoftmax_backward((3, 3)),
		np.array([[-1.4585897 , -5.008274 , 6.4668627 ],
		 [ 2.1793516 , -0.81108296, -1.3682691 ],
		 [ 8.998467 , -5.613649 , -3.3848193 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_op_logsoftmax():
	mugrade.submit(logsoftmax_forward((3, 4)))
	mugrade.submit(logsoftmax_forward((3, 5), mult=1e5))
	mugrade.submit(logsoftmax_forward((3, 6), mult=1e5))
	mugrade.submit(logsoftmax_backward((1, 3)))
	mugrade.submit(logsoftmax_backward((3, 6), mult=1e5))


def test_init_uniform_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.uniform(x, low=-0.42, high=13.37)),
		np.array([[ 4.7449083 , 12.690351 , 9.674196 , 7.8355007 , 1.731497 ],
		 [ 1.7311645 , 0.380973 , 11.5245695 , 7.869376 , 9.344321 ],
		 [-0.13613982, 12.955057 , 11.059384 , 2.5081563 , 2.0873663 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_normal_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.normal(x, mean=13.37, std=4.2)),
		np.array([[15.4562 , 12.789289, 16.090292, 19.766726, 12.386556],
		 [12.386624, 20.002693, 16.593225, 11.398208, 15.648752],
		 [11.423646, 11.413935, 14.386242, 5.334223, 6.125345]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_constant_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.constant(x, c=3)),
		np.array([[3., 3., 3., 3., 3.],
		 [3., 3., 3., 3., 3.],
		 [3., 3., 3., 3., 3.]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_ones_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.ones(x)),
		np.array([[1., 1., 1., 1., 1.],
		 [1., 1., 1., 1., 1.],
		 [1., 1., 1., 1., 1.]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_zeros_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.zeros(x)),
		np.array([[0., 0., 0., 0., 0.],
		 [0., 0., 0., 0., 0.],
		 [0., 0., 0., 0., 0.]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_kaiming_uniform_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.kaiming_uniform(x, mode='fan_in')),
		np.array([[-0.35485414, 1.2748126 , 0.65617794, 0.27904832, -0.9729262 ],
		 [-0.97299445, -1.2499284 , 1.0357026 , 0.28599644, 0.58851814],
		 [-1.3559918 , 1.3291057 , 0.9402898 , -0.81362784, -0.8999349 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_kaiming_uniform_2():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.kaiming_uniform(x, mode='fan_out')),
		np.array([[-0.27486882, 0.98746556, 0.50827324, 0.21614991, -0.7536254 ],
		 [-0.75367826, -0.9681903 , 0.80225176, 0.2215319 , 0.4558642 ],
		 [-1.0503467 , 1.0295209 , 0.72834533, -0.6302334 , -0.6970866 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_kaiming_normal_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.kaiming_normal(x, mode='fan_in')),
		np.array([[ 0.4055654 , -0.11289233, 0.5288355 , 1.2435486 , -0.19118543],
		 [-0.19117202, 1.2894219 , 0.62660784, -0.38332424, 0.4429984 ],
		 [-0.37837896, -0.38026676, 0.19756137, -1.5621868 , -1.4083896 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_kaiming_normal_2():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.kaiming_normal(x, mode='fan_out')),
		np.array([[ 0.31414962, -0.08744602, 0.4096342 , 0.96324867, -0.1480916 ],
		 [-0.14808121, 0.99878186, 0.48536834, -0.29692167, 0.3431451 ],
		 [-0.2930911 , -0.29455337, 0.15303038, -1.2100646 , -1.0909338 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_xavier_uniform_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.xavier_uniform(x, gain=1.5)),
		np.array([[-0.32595432, 1.1709901 , 0.60273796, 0.25632226, -0.8936898 ],
		 [-0.89375246, -1.1481324 , 0.95135355, 0.26270452, 0.54058844],
		 [-1.245558 , 1.2208616 , 0.8637113 , -0.74736494, -0.826643 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_init_xavier_normal_1():
	np.testing.assert_allclose(init_a_tensor_of_shape((3, 5), lambda x: ndl.init.xavier_normal(x, gain=0.33)),
		np.array([[ 0.08195783 , -0.022813609, 0.10686861 , 0.25129992 ,
		 -0.038635306],
		 [-0.038632598, 0.2605701 , 0.12662673 , -0.07746328 ,
		 0.08952241 ],
		 [-0.07646392 , -0.07684541 , 0.039923776, -0.31569123 ,
		 -0.28461143 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_init():
	mugrade.submit(init_a_tensor_of_shape((2, 5), lambda x: ndl.init.kaiming_normal(x, mode='fan_in')))
	mugrade.submit(init_a_tensor_of_shape((2, 5), lambda x: ndl.init.kaiming_normal(x, mode='fan_out')))
	mugrade.submit(init_a_tensor_of_shape((2, 5), lambda x: ndl.init.kaiming_uniform(x, mode='fan_in')))
	mugrade.submit(init_a_tensor_of_shape((2, 5), lambda x: ndl.init.kaiming_uniform(x, mode='fan_out')))
	mugrade.submit(init_a_tensor_of_shape((2, 5), lambda x: ndl.init.xavier_uniform(x, gain=0.33)))
	mugrade.submit(init_a_tensor_of_shape((2, 5), lambda x: ndl.init.xavier_normal(x, gain=1.3)))


def test_nn_linear_weight_init_1():
	np.testing.assert_allclose(nn_linear_weight_init(),
		np.array([[-1.7989244e-01, -2.5801066e-01, -1.6772059e-01, -3.0753542e-02],
		 [-1.3531087e-01, 1.3903665e-02, -1.7995423e-01, 3.5988665e-01],
		 [ 1.7599125e-01, -2.9082534e-01, -8.5967965e-02, 9.7137764e-02],
		 [-2.8342956e-01, 3.6552837e-01, -4.2917967e-02, 2.1888553e-01],
		 [ 2.2233275e-01, -1.0487639e-01, -6.3419461e-02, 6.3693158e-02],
		 [ 1.9667138e-01, -2.3599467e-01, -1.6013059e-01, 1.2867336e-01],
		 [-2.6588942e-04, -2.4297924e-01, -6.5659001e-02, -2.2738703e-01]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_bias_init_1():
	np.testing.assert_allclose(nn_linear_bias_init(),
		np.array([ 0.023962496, 0.25124863 , -0.23792793 , 0.34573108 ],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_forward_1():
	np.testing.assert_allclose(linear_forward((10, 5), (1, 10)),
		np.array([[4.500906 , 6.1882277, 2.776592 , 2.7484004, 4.4740047]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_forward_2():
	np.testing.assert_allclose(linear_forward((10, 5), (3, 10)),
		np.array([[6.0984383, 6.4257445, 1.960106 , 3.021892 , 4.995041 ],
		 [3.9694996, 5.4709086, 3.9861765, 1.3167176, 6.1898117],
		 [4.101536 , 5.3559494, 3.5345604, 1.478467 , 5.5113983]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_forward_3():
	np.testing.assert_allclose(linear_forward((10, 5), (1, 3, 10)),
		np.array([[[4.7056465 , 5.893398 , 3.41159 , 1.711092 , 5.4316854 ],
		 [5.057965 , 5.8713446 , 2.775173 , 2.7890613 , 4.190651 ],
		 [3.4775314 , 5.8348427 , 2.8680677 , 0.35788536, 6.187058 ]]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_backward_1():
	np.testing.assert_allclose(linear_backward((10, 5), (1, 10)),
		np.array([[ 5.2947245 , 0.91959167, 0.26689678, -3.8478894 , -2.229095 ,
		 5.527555 , 1.4721361 , 3.413569 , 0.15164185, 1.9769696 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_backward_2():
	np.testing.assert_allclose(linear_backward((10, 5), (3, 10)),
		np.array([[ 5.9509444 , 1.2286671 , 1.2691393 , -5.139695 ,
		 -1.7781156 , 6.4433284 , 1.3105502 , 3.054307 ,
		 -0.15840921 , 1.6229409 ],
		 [ 3.8984354 , -0.8659983 , -0.16057509 , -2.449396 ,
		 -3.1176708 , 4.395191 , 2.24761 , 5.2684264 ,
		 1.4666293 , 3.4414766 ],
		 [ 4.012814 , -0.47632354 , 0.022940274, -2.7253392 ,
		 -2.7357936 , 4.4877257 , 1.908491 , 4.623176 ,
		 1.0354484 , 2.9248025 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_linear_backward_3():
	np.testing.assert_allclose(linear_backward((10, 5), (1, 3, 10)),
		np.array([[[ 4.5792813 , -0.16904578, 0.15121321, -3.3662672 ,
		 -2.5989559 , 5.0748367 , 1.8369799 , 4.4778666 ,
		 0.7916713 , 2.7237954 ],
		 [ 5.24142 , 1.0139976 , 0.74751204, -3.9791915 ,
		 -2.1211596 , 5.670805 , 0.98158216, 3.211544 ,
		 -0.44029966, 1.6958663 ],
		 [ 3.7483032 , -1.1836054 , -0.93143713, -2.8771822 ,
		 -2.3863206 , 3.960793 , 3.1479712 , 4.927534 ,
		 2.7126913 , 3.3999367 ]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_linear():
	mugrade.submit(linear_forward((3, 5), (1, 3)))
	mugrade.submit(linear_forward((3, 5), (3, 3)))
	mugrade.submit(linear_forward((3, 5), (1, 3, 3)))
	mugrade.submit(linear_backward((4, 5), (1, 4)))
	mugrade.submit(linear_backward((4, 5), (3, 4)))
	mugrade.submit(linear_backward((4, 5), (1, 3, 4)))


def test_nn_relu_forward_1():
	np.testing.assert_allclose(relu_forward(2, 2),
		np.array([[3.35, 4.2 ],
		 [0.25, 4.5 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_relu_backward_1():
	np.testing.assert_allclose(relu_backward(3, 2),
		np.array([[7.5, 2.7],
		 [0.6, 0.2],
		 [0.3, 6.7]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_relu():
	mugrade.submit(relu_forward(2, 3))
	mugrade.submit(relu_backward(3, 4))


def test_nn_sequential_forward_1():
	np.testing.assert_allclose(sequential_forward(batches=3),
		np.array([[ 0.6873627 , -0.027492179, 0.3592106 , -0.7063965 ,
		 -0.028860092],
		 [ 0.56332785 , -0.048734184, 0.22379503 , -0.56091243 ,
		 0.04575953 ],
		 [ 0.6453639 , -0.07970236 , 0.5047174 , -1.0343912 ,
		 0.02541852 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_sequential_backward_1():
	np.testing.assert_allclose(sequential_backward(batches=3),
		np.array([[ 0.13378283 , -0.18285003 , 0.020140318 , 0.0055085868,
		 0.040184163 ],
		 [-0.06074809 , 0.108564146 , 0.080404684 , 0.15420868 ,
		 -0.2055909 ],
		 [ 0.13378283 , -0.18285003 , 0.020140318 , 0.0055085868,
		 0.040184163 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_sequential():
	mugrade.submit(sequential_forward(batches=2))
	mugrade.submit(sequential_backward(batches=2))


def test_nn_softmax_loss_forward_1():
	np.testing.assert_allclose(softmax_loss_forward(5, 10),
		np.array(4.041218, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_softmax_loss_forward_2():
	np.testing.assert_allclose(softmax_loss_forward(3, 11),
		np.array(3.3196716, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_softmax_loss_backward_1():
	np.testing.assert_allclose(softmax_loss_backward(5, 10),
		np.array([[ 0.00068890385, 0.0015331834 , 0.013162163 , -0.16422154 ,
		 0.023983022 , 0.0050903494 , 0.00076135644, 0.050772052 ,
		 0.0062173656 , 0.062013146 ],
		 [ 0.012363418 , 0.02368262 , 0.11730081 , 0.001758993 ,
		 0.004781439 , 0.0029000894 , -0.19815083 , 0.017544521 ,
		 0.015874943 , 0.0019439887 ],
		 [ 0.001219767 , 0.08134181 , 0.057320606 , 0.0008595553 ,
		 0.0030001428 , 0.0009499555 , -0.19633561 , 0.0008176346 ,
		 0.0014898272 , 0.0493363 ],
		 [-0.19886842 , 0.08767337 , 0.017700946 , 0.026406704 ,
		 0.0013147127 , 0.0107361665 , 0.009714483 , 0.023893777 ,
		 0.019562569 , 0.0018656658 ],
		 [ 0.007933789 , 0.017656967 , 0.027691642 , 0.0005605318 ,
		 0.05576411 , 0.0013114461 , 0.06811045 , 0.011835824 ,
		 0.0071787895 , -0.19804356 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_softmax_loss_backward_2():
	np.testing.assert_allclose(softmax_loss_backward(3, 11),
		np.array([[ 0.0027466794, 0.020295369 , 0.012940894 , 0.04748398 ,
		 0.052477922 , 0.090957515 , 0.0028875037, 0.012940894 ,
		 0.040869843 , 0.04748398 , -0.33108455 ],
		 [ 0.0063174255, 0.001721699 , 0.09400159 , 0.0034670753,
		 0.038218185 , 0.009424488 , 0.0042346967, 0.08090791 ,
		 -0.29697907 , 0.0044518122, 0.054234188 ],
		 [ 0.14326698 , 0.002624026 , 0.0032049934, 0.01176007 ,
		 0.045363605 , 0.0043262867, 0.039044812 , 0.017543964 ,
		 0.0037236712, -0.3119051 , 0.04104668 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_softmax_loss():
	mugrade.submit(softmax_loss_forward(4, 9))
	mugrade.submit(softmax_loss_forward(2, 7))
	mugrade.submit(softmax_loss_backward(4, 9))
	mugrade.submit(softmax_loss_backward(2, 7))


def test_nn_layernorm_forward_1():
	np.testing.assert_allclose(layernorm_forward((3, 3, 3, 3), (3,)),
		np.array([[[[-0.502163 , -0.89385015 , 1.3960134 ],
		 [-1.2977618 , 1.1355416 , 0.16222022 ],
		 [ 1.1354151 , 0.16220148 , -1.2976183 ]],
		
		 [[ 1.2159606 , -1.233331 , 0.017371094],
		 [ 0.8113998 , -1.4088044 , 0.5974043 ],
		 [ 1.4126289 , -0.6490463 , -0.76358426 ]],
		
		 [[ 0.7071053 , -1.4142108 , 0.7071053 ],
		 [ 1.4101961 , -0.7970682 , -0.6131292 ],
		 [-1.1401021 , -0.15459011 , 1.2946924 ]]],
		
		
		 [[[-1.3676087 , 0.9956191 , 0.37198952 ],
		 [ 1.172921 , -1.2706645 , 0.097743444],
		 [-0.73761255 , -0.67614484 , 1.4137576 ]],
		
		 [[-0.4777138 , -0.9138872 , 1.3916008 ],
		 [ 0.8931891 , 0.5029609 , -1.3961501 ],
		 [-1.295289 , 1.13923 , 0.156059 ]],
		
		 [[-1.3928715 , 0.9083943 , 0.48447698 ],
		 [ 1.4125699 , -0.76514184 , -0.6474278 ],
		 [-0.91201943 , 1.3920304 , -0.4800102 ]]],
		
		
		 [[[-1.2853236 , 1.1534953 , 0.13182808 ],
		 [ 0.64776504 , -1.4125961 , 0.764831 ],
		 [ 0.080369376, -1.2629472 , 1.182578 ]],
		
		 [[-0.37876505 , -0.9906164 , 1.3693817 ],
		 [-1.39425 , 0.4920883 , 0.90216184 ],
		 [-0.95025164 , 1.3821844 , -0.43193254 ]],
		
		 [[-1.3234963 , 1.093323 , 0.23017325 ],
		 [ 0.40937644 , -1.3769935 , 0.9676171 ],
		 [ 0.8774681 , 0.5217377 , -1.3992065 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_forward_2():
	np.testing.assert_allclose(layernorm_forward((3, 3, 5, 3), (5,3)),
		np.array([[[[ 1.1510044 , 1.2556413 , -0.17439461 ],
		 [-1.4997936 , 0.8370941 , -0.34878922 ],
		 [ 1.430036 , -0.69757843 , 0.48830494 ],
		 [-1.6393093 , -1.4300357 , -0.7673363 ],
		 [ 0.7673362 , 0.55806273 , 0.069757774]],
		
		 [[-1.9048932 , 0.18266058 , 0.57407695 ],
		 [ 0.2696421 , 0.9220025 , 0.6610585 ],
		 [ 0.48709562 , 1.0524746 , 0.9220025 ],
		 [ 1.139456 , -1.6874396 , -0.12177427 ],
		 [-1.7309303 , -0.42620912 , -0.3392278 ]],
		
		 [[ 0.699455 , 0.5878398 , -2.202539 ],
		 [-1.7188733 , 0.2901995 , -0.23067127 ],
		 [ 0.7366602 , -1.1980026 , 0.2901995 ],
		 [ 0.66225016 , 0.77386504 , 0.9598903 ],
		 [-1.0119773 , 0.3274045 , 1.0343007 ]]],
		
		
		 [[[-0.15710208 , -0.70695925 , -1.767398 ],
		 [-0.5498572 , 0.15710188 , 1.3353672 ],
		 [-0.47130623 , 0.549857 , 1.1782653 ],
		 [ 0.5105815 , -0.7855103 , 0.6676836 ],
		 [ 0.549857 , -1.9245 , 1.4139181 ]],
		
		 [[-1.1805742 , 0.23611479 , 0.94445914 ],
		 [-0.5115822 , -0.82640195 , -0.98381186 ],
		 [ 0.3935247 , -1.1805742 , 1.4560413 ],
		 [-0.62963957 , -1.1805742 , 0.94445914 ],
		 [-0.3935247 , 1.4560413 , 1.4560413 ]],
		
		 [[-0.75619316 , -0.9582295 , 1.75483 ],
		 [ 1.75483 , -0.5252945 , -0.8139178 ],
		 [ 1.0909964 , -0.78505546 , -0.8139178 ],
		 [ 0.802373 , 0.94668466 , -0.98709184 ],
		 [-1.0159541 , 0.57147425 , -0.26553345 ]]],
		
		
		 [[[-0.6701204 , -0.99049664 , 0.7715726 ],
		 [-0.7902615 , -0.10946197 , -0.3497442 ],
		 [-1.1506848 , -0.5499793 , 1.8127953 ],
		 [ 0.8116196 , -1.511108 , 0.13082017 ],
		 [ 1.5725133 , 1.332231 , -0.30969712 ]],
		
		 [[-0.8353074 , -0.7734328 , -0.8662447 ],
		 [-1.1137433 , 0.74249554 , 1.7634268 ],
		 [-0.21656114 , 0.37124786 , -1.2374924 ],
		 [ 0.092812 , -1.2065551 , -0.30937305 ],
		 [ 0.8971821 , 0.9281194 , 1.7634268 ]],
		
		 [[ 0.92056584 , 0.37418684 , 0.9702366 ],
		 [-2.208695 , 0.8212241 , -0.1225212 ],
		 [ 0.8212241 , 0.12583283 , -1.413962 ],
		 [-1.8113284 , 1.1689199 , 0.22517458 ],
		 [-0.22186272 , 0.12583283 , 0.22517458 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_forward_3():
	np.testing.assert_allclose(layernorm_forward((3, 3, 3, 3), (3,3,3)),
		np.array([[[[-0.9616619 , -1.450147 , 1.4056128 ],
		 [-1.299844 , 0.2031875 , -0.39802507 ],
		 [-0.585904 , -0.7362072 , -0.9616619 ]],
		
		 [[ 1.1425823 , -0.6234797 , 0.27833918 ],
		 [ 1.2553095 , -1.8634808 , 0.9547032 ],
		 [ 1.5183402 , 0.841976 , 0.8044 ]],
		
		 [[ 1.2177339 , -1.299844 , 1.2177339 ],
		 [ 0.7292485 , -0.17257038 , -0.097418696],
		 [-1.1119651 , -0.47317666 , 0.4662181 ]]],
		
		
		 [[[-1.3985255 , 0.82629746 , 0.23919137 ],
		 [-0.6878182 , -1.4603262 , -1.0277218 ],
		 [-1.1204226 , -1.0895224 , -0.03891154 ]],
		
		 [[-1.058622 , -1.2749243 , -0.1316124 ],
		 [ 1.2589018 , 0.79539716 , -1.4603262 ],
		 [-0.34791467 , 1.2589018 , 0.6099953 ]],
		
		 [[-0.7805192 , 1.5679051 , 1.1353006 ],
		 [ 1.1971012 , 0.053789474, 0.11559006 ],
		 [ 0.5481946 , 1.5370048 , 0.73359644 ]]],
		
		
		 [[[-0.9806486 , 1.3914285 , 0.39772066 ],
		 [ 1.3914285 , -1.42942 , 1.5517039 ],
		 [-0.11516102 , -1.3653097 , 0.9106022 ]],
		
		 [[-0.4998221 , -1.1729791 , 1.4234837 ],
		 [-1.461475 , 0.013059404, 0.3336104 ],
		 [-1.3973649 , 0.3336104 , -1.0127037 ]],
		
		 [[-0.91653836 , 0.87854713 , 0.23744518 ],
		 [ 1.0067674 , -0.53187716 , 1.4875939 ],
		 [ 0.23744518 , 0.07716969 , -0.78831804 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_forward_4():
	np.testing.assert_allclose(layernorm_forward((3, 3, 4, 3), (4,3)),
		np.array([[[[-1.3167651 , -1.5037612 , -0.3817839 ],
		 [ 1.4881784 , 0.335035 , 0.6466954 ],
		 [-1.0362706 , 0.9583557 , -0.25711975 ],
		 [ 0.45969915 , -0.84927446 , 1.4570123 ]],
		
		 [[-1.3168392 , -1.5961688 , 0.23942527 ],
		 [-0.359138 , 1.9154025 , 1.0375097 ],
		 [-1.0375097 , 0.5586589 , -0.5985634 ],
		 [-0.23942547 , 0.63846743 , 0.75817996 ]],
		
		 [[-0.85353833 , 0.9334441 , 0.84627426 ],
		 [-2.0739162 , -0.984293 , 0.49759474 ],
		 [ 0.7155194 , -0.8971232 , -0.5484437 ],
		 [ 0.36683986 , 1.4564632 , 0.54117966 ]]],
		
		
		 [[[ 1.1585436 , -0.23170853 , -0.07005131 ],
		 [-0.74901164 , 1.481858 , 0.77056617 ],
		 [-0.26404008 , -1.4926348 , -1.363309 ],
		 [-1.1046575 , 0.67357177 , 1.1908748 ]],
		
		 [[-0.14616801 , -0.61151916 , 0.71294224 ],
		 [ 1.1424972 , -1.5064255 , 0.7487383 ],
		 [ 0.8203307 , -1.8285917 , 0.8919235 ],
		 [ 1.0709047 , -0.3609455 , -0.9336855 ]],
		
		 [[-0.8838127 , 0.36852676 , 0.6816117 ],
		 [-0.2967785 , -1.1577619 , 0.32939118 ],
		 [ 1.6600019 , -1.35344 , 0.40766233 ],
		 [ 1.542595 , 0.09457758 , -1.3925756 ]]],
		
		
		 [[[ 0.7961765 , 1.1861404 , -0.5362003 ],
		 [ 1.1536436 , 0.6661886 , 0.95866144 ],
		 [-0.8936672 , -0.30872124 , 0.8936675 ],
		 [-1.1861402 , -1.6410981 , -1.0886492 ]],
		
		 [[ 1.2168632 , 0.5531198 , -0.44249573 ],
		 [ 1.6435556 , -0.015803402 , 0.3160685 ],
		 [-1.1062392 , -1.8648032 , 1.1220427 ],
		 [-0.91659814 , -0.63213664 , 0.12642744 ]],
		
		 [[ 0.45330352 , 1.5023772 , -0.0129513275],
		 [-1.3340069 , -0.4403516 , 0.7641402 ],
		 [-1.139734 , 0.72528565 , -1.0231704 ],
		 [ 0.7641402 , -1.4505706 , 1.1915405 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_backward_1():
	np.testing.assert_allclose(layernorm_backward((3, 3, 3, 3), (3,)),
		np.array([[[[-6.91413879e-06, -1.16825104e-05, 1.82390213e-05],
		 [-1.41024590e-04, 1.23143196e-04, 1.77621841e-05],
		 [ 7.82823563e-03, 1.15013123e-03, -8.97932053e-03]],
		
		 [[ 8.43182206e-05, -8.25747848e-05, -1.73598528e-06],
		 [ 7.39097595e-06, -1.29938126e-05, 5.72204590e-06],
		 [ 1.02472305e-03, -4.68254089e-04, -5.55515289e-04]],
		
		 [[ 1.06096268e-05, -2.13384628e-05, 1.06096268e-05],
		 [ 5.32150269e-04, -3.03268433e-04, -2.29358673e-04],
		 [-1.06692314e-04, -1.50203705e-05, 1.21712685e-04]]],
		
		
		 [[[-2.36034393e-05, 1.71661377e-05, 6.43730164e-06],
		 [ 5.23924828e-04, -5.68032265e-04, 4.41074371e-05],
		 [-8.22544098e-05, -7.51018524e-05, 1.58071518e-04]],
		
		 [[-5.48362732e-05, -1.06334686e-04, 1.60932541e-04],
		 [ 7.27176666e-06, 4.29153442e-06, -1.18017197e-05],
		 [-6.48498535e-05, 5.72204590e-05, 7.56978989e-06]],
		
		 [[-1.77621841e-05, 1.16825104e-05, 6.31809235e-06],
		 [ 1.38282776e-04, -7.43865967e-05, -6.34193420e-05],
		 [-1.61409378e-04, 2.49624252e-04, -8.77380371e-05]]],
		
		
		 [[[-2.26497650e-05, 1.97887421e-05, 2.86102295e-06],
		 [ 3.45706940e-06, -7.74860382e-06, 4.17232513e-06],
		 [ 1.16229057e-06, -2.46465206e-05, 2.35140324e-05]],
		
		 [[-4.64916229e-06, -1.10864639e-05, 1.58548355e-05],
		 [-4.67300415e-05, 1.64508820e-05, 3.05175781e-05],
		 [-3.67164612e-05, 5.38825989e-05, -1.70469284e-05]],
		
		 [[-5.11407852e-05, 4.18424606e-05, 9.41753387e-06],
		 [ 1.00135803e-05, -3.42130661e-05, 2.41994858e-05],
		 [ 1.50203705e-04, 9.15527344e-05, -2.41994858e-04]]]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_backward_2():
	np.testing.assert_allclose(layernorm_backward((3, 3, 5, 3), (5,3)),
		np.array([[[[ -0.7102292 , 0.043504417, 1.5481405 ],
		 [ -1.3224652 , -1.7822418 , 2.3034925 ],
		 [ 1.8206737 , 3.1926098 , -1.3761859 ],
		 [ -3.5142558 , -0.4126895 , 3.2226734 ],
		 [ -1.8146923 , -1.5596921 , 0.3613583 ]],
		
		 [[ -5.7492414 , 1.3648282 , -1.1774058 ],
		 [ 0.70532465 , -1.9347138 , -1.5370817 ],
		 [ -0.7270615 , -1.6652436 , -1.9347138 ],
		 [ -1.280673 , -0.18393373 , 3.8101208 ],
		 [ -1.1568713 , 6.0198097 , 5.446856 ]],
		
		 [[ -1.4102125 , -0.96929026 , -12.005377 ],
		 [ 0.9773669 , 0.77843547 , 4.6584377 ],
		 [ -1.5238328 , 6.9860377 , 0.77843547 ],
		 [ -1.279304 , -1.6192408 , -1.7909446 ],
		 [ 7.5942445 , 0.5252049 , -1.699965 ]]],
		
		
		 [[[ 2.388496 , 5.183607 , -3.5446868 ],
		 [ 4.659123 , 0.18822765 , -0.68440247 ],
		 [ 4.296371 , -2.0823998 , -1.9142809 ],
		 [ -1.908448 , 5.327063 , -2.5037463 ],
		 [ -2.0823998 , -7.4814944 , 0.1589694 ]],
		
		 [[ -0.44978845 , -1.942868 , -2.6894088 ],
		 [ 1.1380314 , 1.2748996 , 0.7999946 ],
		 [ -2.538526 , -0.44978845 , 1.9517576 ],
		 [ 1.3332502 , -0.44978845 , -2.6894088 ],
		 [ 0.80812895 , 1.9517576 , 1.9517576 ]],
		
		 [[ 0.72957575 , 0.51221 , 4.0667353 ],
		 [ 4.0667353 , 0.46103168 , 0.71610296 ],
		 [ -2.7319665 , 0.72736967 , 0.71610296 ],
		 [ -3.372253 , -3.1886785 , 0.43958747 ],
		 [ 0.35557353 , -3.2017717 , -0.29635346 ]]],
		
		
		 [[[ 1.926955 , 1.7956808 , -4.717992 ],
		 [ 2.0666192 , -0.6445155 , 0.7360668 ],
		 [ 1.0366971 , 1.6013618 , 6.337797 ],
		 [ -4.7289906 , -2.866608 , -2.1465812 ],
		 [ 1.223458 , -2.1456711 , 0.5257231 ]],
		
		 [[ 1.4620945 , 1.4687774 , 1.4411647 ],
		 [ 0.793555 , -3.495454 , 4.266738 ],
		 [ -0.027669787, -2.6376524 , 0.10390782 ],
		 [ -1.4541107 , 0.30160403 , 0.36024833 ],
		 [ -3.4479668 , -3.401973 , 4.266738 ]],
		
		 [[ -3.3839073 , -0.056337357, -3.4199486 ],
		 [-13.679265 , -3.152297 , 5.3821077 ],
		 [ -3.152297 , 2.5703626 , 8.856443 ],
		 [ 0.99810123 , -2.9641733 , 1.4770229 ],
		 [ 6.4768085 , 2.5703626 , 1.4770229 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_backward_3():
	np.testing.assert_allclose(layernorm_backward((3, 3, 3, 3), (3,3,3)),
		np.array([[[[ 2.4946537 , -1.4994018 , 1.402458 ],
		 [ 0.2966687 , -0.7675037 , 2.09426 ],
		 [ 2.640616 , 2.814868 , 2.4946537 ]],
		
		 [[-1.1158564 , 2.7089527 , -1.1124656 ],
		 [-0.23030263 , -9.669523 , -2.0225945 ],
		 [ 2.999534 , -2.2672284 , -2.3046134 ]],
		
		 [[-0.5561779 , 0.2966687 , -0.5561779 ],
		 [-2.318885 , 1.1146638 , 0.74276626 ],
		 [ 1.8041375 , 2.3499095 , -1.834079 ]]],
		
		
		 [[[-1.18989 , -1.8494833 , -0.88383377 ],
		 [ 1.9512926 , -1.8816091 , 1.4192299 ],
		 [ 0.99297935 , 1.1503466 , 0.18419936 ]],
		
		 [[ 1.2922851 , -0.040628135, 0.54604495 ],
		 [-0.026294885, -1.8777221 , -1.8816091 ],
		 [ 1.3046929 , -0.026294885, -1.8259168 ]],
		
		 [[ 1.9475832 , 3.3454206 , -0.85126776 ],
		 [-0.47268838 , -0.1826064 , -0.42406368 ],
		 [-1.7348694 , 2.9155476 , -1.900844 ]]],
		
		
		 [[[ 1.7070836 , 0.7168933 , -1.7081227 ],
		 [ 0.7168933 , -1.4124202 , 2.6925936 ],
		 [ 0.35730192 , -0.72802055 , -2.1638258 ]],
		
		 [[ 1.714014 , 0.8233391 , 1.0660115 ],
		 [-1.7882642 , -0.19644421 , -1.4954054 ],
		 [-1.059174 , -1.4954054 , 1.6015092 ]],
		
		 [[ 1.8722271 , -2.2217805 , -1.1380395 ],
		 [-1.9015547 , 1.7877876 , 1.8322664 ],
		 [-1.1380395 , -0.47410595 , 2.032682 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_layernorm_backward_4():
	np.testing.assert_allclose(layernorm_backward((3, 3, 4, 3), (4,3)),
		np.array([[[[ -0.17698807 , -2.1739192 , 1.4319632 ],
		 [ 1.8987193 , -1.3598061 , -2.094149 ],
		 [ 1.5574701 , -1.8887997 , 1.0023668 ],
		 [ -1.737323 , 2.015798 , 1.5246673 ]],
		
		 [[ 1.4835719 , -2.2752836 , -1.9558562 ],
		 [ 1.9958774 , 8.831828 , -3.959023 ],
		 [ 3.2744198 , -3.652888 , 3.1165173 ],
		 [ 1.2712535 , -3.931098 , -4.1993203 ]],
		
		 [[ 6.2995124 , -2.6672902 , -2.708393 ],
		 [-13.094049 , 6.1648917 , -1.66607 ],
		 [ -2.5221329 , 6.2908697 , 5.5072145 ],
		 [ -0.90126944 , 1.1804776 , -1.8837616 ]]],
		
		
		 [[[ -0.74241406 , 1.0235933 , 0.37810436 ],
		 [ 2.134577 , 2.2985034 , -1.9568076 ],
		 [ 1.143513 , -2.2669034 , -0.7606674 ],
		 [ 1.2237964 , -1.9437418 , -0.53155285 ]],
		
		 [[ 2.1548872 , 3.9437861 , -1.2934985 ],
		 [ -0.30813932 , -0.508698 , -1.3165002 ],
		 [ -1.312262 , -6.5429835 , -1.235779 ],
		 [ -0.6871276 , 3.1529336 , 3.953382 ]],
		
		 [[ 2.8252783 , -2.3913789 , -3.4403226 ],
		 [ 1.3729417 , 1.7762549 , -2.200725 ],
		 [ 4.0031385 , 0.050166845, -2.57143 ],
		 [ 1.8806379 , -0.8971944 , -0.407367 ]]],
		
		
		 [[[ -1.0746119 , 0.498784 , 2.177073 ],
		 [ 0.27296203 , -1.1337051 , -0.7016352 ],
		 [ 2.0543156 , 1.6537583 , -0.8944543 ],
		 [ 0.6610789 , -4.795637 , 1.2820721 ]],
		
		 [[ -2.6950643 , -3.5015779 , 3.6067595 ],
		 [ 3.8473694 , 0.4730256 , -2.1001296 ],
		 [ 4.1865973 , -9.119325 , -3.4020064 ],
		 [ 4.861633 , 4.516126 , -0.6734073 ]],
		
		 [[ -1.4679258 , 3.7487893 , 0.48006263 ],
		 [ -0.55941486 , 2.2657197 , -1.8622265 ],
		 [ 1.2854738 , -1.8767545 , 1.9985733 ],
		 [ -1.8622265 , -2.108315 , -0.04175523 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_layernorm():
	mugrade.submit(layernorm_forward((2, 2, 2, 2), (2,)))
	mugrade.submit(layernorm_forward((2, 2, 4, 2), (4,2)))
	mugrade.submit(layernorm_forward((2, 2, 2, 2), (2,2,2)))
	mugrade.submit(layernorm_forward((2, 2, 3, 2), (3,2)))
	mugrade.submit(layernorm_backward((2, 2, 2, 2), (2,)))
	mugrade.submit(layernorm_backward((2, 2, 4, 2), (4,2)))
	mugrade.submit(layernorm_backward((2, 2, 2, 2), (2,2,2)))
	mugrade.submit(layernorm_backward((2, 2, 3, 2), (3,2)))


def test_nn_batchnorm_check_model_eval_switches_training_flag_1():
	np.testing.assert_allclose(check_training_mode(),
		np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		 0, 0, 0, 0, 0]), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_1():
	np.testing.assert_allclose(batchnorm_forward(4, 4),
		np.array([[ 7.8712696e-01, -3.1676728e-01, -6.4885163e-01, 2.0828949e-01],
		 [-7.9508079e-03, 1.0092355e+00, 1.6221288e+00, 8.5209310e-01],
		 [ 8.5073310e-01, -1.4954363e+00, -9.6686421e-08, -1.6852506e+00],
		 [-1.6299094e+00, 8.0296844e-01, -9.7327745e-01, 6.2486827e-01]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_2():
	np.testing.assert_allclose(batchnorm_forward(4, 4, 4),
		np.array([[[ 1.7091177 , 0.6692038 , 0.014443393, -0.52477115 ],
		 [ 0.49391493 , 0.2837384 , 0.914268 , -1.271568 ],
		 [-0.479005 , 1.5880705 , 0.3160241 , -0.19279458 ],
		 [ 1.094244 , 1.3526633 , 0.092869386, 0.4804982 ]],
		
		 [[-1.0254704 , 1.6320871 , 1.2084184 , 1.516541 ],
		 [ 0.32577386 , -0.80917954 , 1.6709037 , 0.7040915 ],
		 [ 1.1110532 , 1.4926673 , 0.9520474 , 1.3654624 ],
		 [-0.10094502 , 0.9973367 , 1.0296392 , 0.9973367 ]],
		
		 [[-0.52477115 , -0.5632864 , -1.3721082 , -1.0254704 ],
		 [ 0.2837384 , 0.49391493 , -2.112274 , 0.4098444 ],
		 [-1.242233 , -1.1468295 , -0.6062097 , -1.2104319 ],
		 [-0.94080764 , -1.1346221 , -1.7806703 , 0.35128856 ]],
		
		 [[-0.8714091 , 0.05295868 , -0.13961793 , -0.7558631 ],
		 [-0.30475575 , -0.6410383 , -1.5658151 , 1.1244446 ],
		 [-1.1468295 , 0.28422296 , -0.22459571 , -0.860619 ],
		 [-1.5545533 , -0.68238837 , -0.90850526 , 0.70661515 ]]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_3():
	np.testing.assert_allclose(batchnorm_forward(4, 3, 4, 2),
		np.array([[[[ 1.6747193 , 1.8831093 ],
		 [-1.1385484 , -0.68703634 ],
		 [-1.416402 , 1.4315976 ],
		 [-0.33971933 , 0.49384153 ]],
		
		 [[-1.0508223 , 1.3257205 ],
		 [ 1.1838375 , -0.022169411],
		 [ 0.8291295 , -0.27046487 ],
		 [-1.1217638 , 0.26159695 ]],
		
		 [[ 1.4945896 , 1.281868 ],
		 [ 0.040993318, -1.3771492 ],
		 [ 1.1046002 , 0.53734326 ],
		 [-1.4126028 , -1.1998814 ]]],
		
		
		 [[[-0.7912314 , -1.3816704 ],
		 [-0.8606948 , 0.007597692],
		 [ 0.5980365 , 0.8411585 ],
		 [ 1.6399876 , 1.3621339 ]],
		
		 [[ 0.5808341 , -1.1572345 ],
		 [ 0.22612621 , -1.6538256 ],
		 [ 0.15518452 , 0.758188 ],
		 [ 0.8646003 , 0.8646003 ]],
		
		 [[-1.3416957 , 1.6364037 ],
		 [-0.70353144 , -0.52626365 ],
		 [-1.1644279 , -0.27808878 ],
		 [ 0.89187884 , 0.75006455 ]]],
		
		
		 [[[-1.416402 , 0.11179286 ],
		 [-0.68703634 , 0.32018304 ],
		 [-1.1732801 , -0.8606948 ],
		 [-0.9648899 , -0.061865643]],
		
		 [[ 1.4321331 , -1.6538256 ],
		 [ 1.1838375 , 1.6449575 ],
		 [-1.2281761 , -0.48328957 ],
		 [-1.4410009 , 0.40348014 ]],
		
		 [[ 0.9627859 , 1.0336932 ],
		 [-0.9517064 , 1.1755073 ],
		 [-1.2353349 , -0.66807795 ],
		 [ 1.5300429 , 0.3600754 ]]],
		
		
		 [[[ 0.5980365 , 1.5010608 ],
		 [-0.8954265 , -0.37445098 ],
		 [ 0.6675 , -0.13132915 ],
		 [-0.65230465 , 0.7022317 ]],
		
		 [[-1.086293 , -1.4410009 ],
		 [ 0.119713776, 0.15518452 ],
		 [-1.1217638 , -0.057640165],
		 [ 1.0419543 , 0.758188 ]],
		
		 [[-0.66807795 , -1.1644279 ],
		 [-0.24263518 , -0.1362745 ],
		 [-0.98716 , 1.0336932 ],
		 [ 0.6437038 , -0.41990298 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_4():
	np.testing.assert_allclose(batchnorm_forward(4, 3, 2, 1),
		np.array([[[[ 0.494267 ],
		 [ 0.76083815]],
		
		 [[-0.4380846 ],
		 [ 1.4247615 ]],
		
		 [[ 1.7774435 ],
		 [ 0.707526 ]]],
		
		
		 [[[-0.83858794],
		 [-0.5275885 ]],
		
		 [[-1.4484419 ],
		 [ 0.66699374]],
		
		 [[ 0.56947213],
		 [-1.5013361 ]]],
		
		
		 [[[ 0.76083815],
		 [ 0.3609818 ]],
		
		 [[ 0.28810966],
		 [-1.5431628 ]],
		
		 [[ 0.50044525],
		 [-0.43141845]]],
		
		
		 [[[ 1.0718377 ],
		 [-2.082586 ]],
		
		 [[ 0.85643566],
		 [ 0.19338876]],
		
		 [[-0.7420396 ],
		 [-0.8800936 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_affine_1():
	np.testing.assert_allclose(batchnorm_forward(4, 4, affine=True),
		np.array([[ 7.49529 , 0.047213316, 2.690084 , 5.5227957 ],
		 [ 4.116209 , 3.8263211 , 7.79979 , 7.293256 ],
		 [ 7.765616 , -3.3119934 , 4.15 , 0.31556034 ],
		 [-2.7771149 , 3.23846 , 1.9601259 , 6.6683874 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_affine_2():
	np.testing.assert_allclose(batchnorm_forward(4, 4, 4, affine=True),
		np.array([[[11.41375 , 6.994116 , 4.2113843 , 1.9197228 ],
		 [ 2.3576574 , 1.7586544 , 3.5556638 , -2.673968 ],
		 [ 3.072239 , 7.723159 , 4.8610544 , 3.7162123 ],
		 [ 7.9591703 , 8.669824 , 5.2053905 , 6.27137 ]],
		
		 [[-0.20824862 , 11.08637 , 9.285778 , 10.595299 ],
		 [ 1.8784554 , -1.3561616 , 5.7120748 , 2.9566607 ],
		 [ 6.64987 , 7.508501 , 6.2921066 , 7.222291 ],
		 [ 4.672401 , 7.6926756 , 7.7815075 , 7.6926756 ]],
		
		 [[ 1.9197228 , 1.7560327 , -1.6814599 , -0.20824862 ],
		 [ 1.7586544 , 2.3576574 , -5.0699806 , 2.1180565 ],
		 [ 1.3549759 , 1.5696337 , 2.7860284 , 1.4265282 ],
		 [ 2.3627787 , 1.8297889 , 0.053156376, 5.9160433 ]],
		
		 [[ 0.44651127 , 4.3750744 , 3.556624 , 0.937582 ],
		 [ 0.08144611 , -0.876959 , -3.512573 , 4.154667 ],
		 [ 1.5696337 , 4.7895017 , 3.6446598 , 2.2136073 ],
		 [ 0.67497826 , 3.0734315 , 2.4516103 , 6.8931913 ]]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_affine_3():
	np.testing.assert_allclose(batchnorm_forward(4, 3, 4, 2, affine=True),
		np.array([[[[ 7.296462 , 8.067505 ],
		 [-3.112629 , -1.4420344 ],
		 [-4.1406875 , 6.3969107 ],
		 [-0.15696144, 2.9272137 ]],
		
		 [[ 1.4974588 , 1.616286 ],
		 [ 1.6091918 , 1.5488914 ],
		 [ 1.5914564 , 1.5364767 ],
		 [ 1.4939117 , 1.5630798 ]],
		
		 [[ 4.074188 , 4.04228 ],
		 [ 3.856149 , 3.6434276 ],
		 [ 4.01569 , 3.9306014 ],
		 [ 3.6381094 , 3.6700177 ]]],
		
		
		 [[[-1.8275563 , -4.0121803 ],
		 [-2.084571 , 1.1281115 ],
		 [ 3.312735 , 4.2122865 ],
		 [ 7.1679535 , 6.1398954 ]],
		
		 [[ 1.5790416 , 1.4921383 ],
		 [ 1.5613062 , 1.4673086 ],
		 [ 1.5577592 , 1.5879093 ],
		 [ 1.59323 , 1.59323 ]],
		
		 [[ 3.6487455 , 4.0954604 ],
		 [ 3.7444701 , 3.7710605 ],
		 [ 3.6753356 , 3.8082867 ],
		 [ 3.9837818 , 3.9625096 ]]],
		
		
		 [[[-4.1406875 , 1.5136336 ],
		 [-1.4420344 , 2.2846773 ],
		 [-3.2411366 , -2.084571 ],
		 [-2.4700928 , 0.87109715]],
		
		 [[ 1.6216066 , 1.4673086 ],
		 [ 1.6091918 , 1.6322478 ],
		 [ 1.4885912 , 1.5258355 ],
		 [ 1.4779499 , 1.570174 ]],
		
		 [[ 3.994418 , 4.005054 ],
		 [ 3.707244 , 4.026326 ],
		 [ 3.6646996 , 3.7497883 ],
		 [ 4.0795064 , 3.9040112 ]]],
		
		
		 [[[ 3.312735 , 6.6539254 ],
		 [-2.213078 , -0.2854687 ],
		 [ 3.5697503 , 0.6140822 ],
		 [-1.3135272 , 3.6982574 ]],
		
		 [[ 1.4956853 , 1.4779499 ],
		 [ 1.5559857 , 1.5577592 ],
		 [ 1.4939117 , 1.547118 ],
		 [ 1.6020976 , 1.5879093 ]],
		
		 [[ 3.7497883 , 3.6753356 ],
		 [ 3.8136046 , 3.8295586 ],
		 [ 3.701926 , 4.005054 ],
		 [ 3.9465554 , 3.7870145 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_forward_affine_4():
	np.testing.assert_allclose(batchnorm_forward(5, 4, 2, 3, affine=True),
		np.array([[[[-1.6007204 , 8.666571 , -1.6007204 ],
		 [-0.20063496, 12.711262 , -0.04507017]],
		
		 [[ 2.8502686 , 2.1917596 , 4.073213 ],
		 [ 5.9546676 , 3.7909958 , -0.4422757 ]],
		
		 [[ 1.114613 , 5.4237795 , 0.87069774],
		 [ 5.342474 , 3.1472387 , 6.39944 ]],
		
		 [[ 0.621593 , 7.0738754 , 1.0696683 ],
		 [ 5.4608054 , 6.715416 , 0.26313305]]],
		
		
		 [[[ 9.599961 , 1.355015 , -0.511765 ],
		 [-1.1340256 , 5.399706 , 5.866401 ]],
		
		 [[-2.6059475 , -0.81856626, -1.4770751 ],
		 [-2.6059475 , -2.9822383 , 2.0036144 ]],
		
		 [[ 5.5050845 , 0.78939295, 1.1959181 ],
		 [ 3.7163737 , 4.7733393 , 1.3585281 ]],
		
		 [[ 5.8192654 , 1.9658186 , 4.9231153 ],
		 [ 8.776562 , 7.4323363 , 3.3100443 ]]],
		
		
		 [[[ 1.8217103 , 8.977701 , 9.444396 ],
		 [ 7.888746 , 3.3773603 , 0.26605988]],
		
		 [[ 2.1917596 , 3.7909958 , -1.0067117 ],
		 [ 0.31030583, 0.78066933, -0.81856626]],
		
		 [[ 5.0985594 , 3.960289 , 7.293795 ],
		 [ 4.5294237 , 4.2042036 , 2.2528834 ]],
		
		 [[ 4.3854246 , 2.682739 , 6.6258006 ],
		 [ 8.238872 , 7.970026 , 7.8804116 ]]],
		
		
		 [[[ 5.088576 , 1.8217103 , 4.6218805 ],
		 [ 8.199877 , 3.0662303 , 1.355015 ]],
		
		 [[-1.6652205 , 6.1428127 , 6.1428127 ],
		 [ 3.2265592 , -1.3830025 , -1.8533659 ]],
		
		 [[ 4.5294237 , 2.6594086 , 7.0498796 ],
		 [ 5.017254 , 1.683748 , 3.878984 ]],
		
		 [[ 1.6073585 , 1.8762038 , 2.9515839 ],
		 [ 4.9231153 , 8.597331 , 3.9373493 ]]],
		
		
		 [[[ 2.9106655 , -0.8228955 , 7.5776167 ],
		 [12.555698 , 7.5776167 , 0.26605988]],
		
		 [[-2.6059475 , 4.6376495 , 2.2858322 ],
		 [ 2.1917596 , -2.5118747 , -1.2889297 ]],
		
		 [[ 7.293795 , 8.025539 , 6.8059645 ],
		 [ 0.7080879 , 7.7003202 , 2.1715784 ]],
		
		 [[ 0.9800534 , 2.861969 , 4.6542697 ],
		 [ 8.059641 , 8.866177 , 7.970026 ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_backward_1():
	np.testing.assert_allclose(batchnorm_backward(5, 4),
		np.array([[ 2.1338463e-04, 5.2094460e-06, -2.8359889e-05, -4.4368207e-06],
		 [-3.8480759e-04, -4.0292739e-06, 1.8370152e-05, -1.1172146e-05],
		 [ 2.5629997e-04, -1.1003018e-05, -9.0479853e-06, 5.5171549e-06],
		 [-4.2676926e-04, 3.4213067e-06, 1.3601780e-05, 1.0166317e-05],
		 [ 3.4189224e-04, 6.4015389e-06, 5.4359434e-06, -7.4505806e-08]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_backward_2():
	np.testing.assert_allclose(batchnorm_backward(5, 3, 4),
		np.array([[[-8.32614023e-06, -1.11871632e-05, -6.53800089e-06,
		 -1.28560932e-05],
		 [ 1.52520834e-05, -1.55642624e-06, 2.07945709e-06,
		 -1.31197276e-05],
		 [ 8.82148743e-06, 5.36441803e-06, -9.89437103e-06,
		 5.60283661e-06]],
		
		 [[ 6.33660238e-06, 4.19083517e-06, -1.24808867e-06,
		 5.80912456e-08],
		 [-1.03779139e-05, -1.03779139e-05, -4.47705406e-06,
		 8.57636314e-06],
		 [ 6.79492950e-06, -8.34465027e-06, 9.68575478e-07,
		 -1.12056732e-05]],
		
		 [[-1.76962931e-06, 1.05089275e-05, 9.43604391e-06,
		 1.05089275e-05],
		 [-7.87451881e-06, 5.65573555e-06, 1.40599905e-05,
		 4.25428141e-07],
		 [ 3.93390656e-06, -1.38282776e-05, 2.74181366e-06,
		 1.14440918e-05]],
		
		 [[-4.65777703e-07, -6.53800089e-06, -4.33262903e-06,
		 1.35961454e-06],
		 [ 4.10601479e-06, 6.43059593e-06, -6.32479805e-06,
		 -5.19230980e-06],
		 [-9.89437103e-06, -1.33514404e-05, 8.58306885e-06,
		 4.17232513e-06]],
		
		 [[ 7.29027670e-06, 4.48885839e-06, -9.63744242e-06,
		 8.72078817e-06],
		 [ 1.31063161e-05, -7.87451881e-06, 7.74189812e-06,
		 -1.02587046e-05],
		 [ 7.51018524e-06, 4.17232513e-06, -1.05798244e-06,
		 -2.53319740e-06]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_backward_3():
	np.testing.assert_allclose(batchnorm_backward(2, 4, 2, 3),
		np.array([[[[ 6.67776912e-05, 3.73087823e-06, -3.59807163e-05],
		 [-6.86440617e-05, -1.72648579e-05, -2.90665776e-05]],
		
		 [[ 7.52555206e-06, -9.28295776e-06, -7.97165558e-06],
		 [ 1.28899701e-05, 6.93835318e-08, -5.11063263e-06]],
		
		 [[ 4.39087535e-06, 5.34454966e-06, 7.99695670e-07],
		 [ 2.18550372e-06, -2.88089109e-06, 5.10613108e-06]],
		
		 [[ 5.37435199e-06, 6.56644488e-06, 1.55965483e-06],
		 [-2.25504232e-06, 6.08960772e-06, 2.87095713e-06]]],
		
		
		 [[[ 1.07493252e-05, 3.86442989e-05, -9.55536962e-07],
		 [ 2.93459743e-05, -3.83649021e-05, 4.10284847e-05]],
		
		 [[ 1.26515515e-05, 1.24131329e-05, -2.84565613e-06],
		 [-3.20328400e-06, -9.99821350e-06, -7.13719055e-06]],
		
		 [[ 5.61277091e-07, 1.14242232e-06, -5.98033284e-06],
		 [ 9.33806120e-07, -5.98033284e-06, -5.62270498e-06]],
		
		 [[-3.80476308e-06, -6.42736768e-06, -2.79148412e-06],
		 [-3.62594915e-06, -6.42736768e-06, 2.87095713e-06]]]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_backward_affine_1():
	np.testing.assert_allclose(batchnorm_backward(5, 4, affine=True),
		np.array([[ 3.8604736e-03, 4.2676926e-05, -1.4114380e-04, -3.2424927e-05],
		 [-6.9427490e-03, -3.3140182e-05, 9.1552734e-05, -8.5830688e-05],
		 [ 4.6386719e-03, -8.9883804e-05, -4.5776367e-05, 4.3869019e-05],
		 [-7.7133179e-03, 2.7418137e-05, 6.6757202e-05, 7.4386597e-05],
		 [ 6.1874390e-03, 5.2213669e-05, 2.8610229e-05, -1.9073486e-06]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_backward_affine_2():
	np.testing.assert_allclose(batchnorm_backward(5, 3, 4, affine=True),
		np.array([[[-1.1634827e-04, -1.5258789e-04, -8.7738037e-05, -1.7547607e-04],
		 [ 5.2154064e-08, 0.0000000e+00, 2.2351742e-08, -2.2351742e-08],
		 [ 5.9604645e-08, 5.9604645e-08, 0.0000000e+00, -5.9604645e-08]],
		
		 [[ 8.7738037e-05, 6.0081482e-05, -1.7166138e-05, 1.4305115e-06],
		 [-2.2351742e-08, -2.2351742e-08, 0.0000000e+00, 3.7252903e-08],
		 [ 5.9604645e-08, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],
		
		 [[-2.4318695e-05, 1.4495850e-04, 1.2779236e-04, 1.4495850e-04],
		 [-1.4901161e-08, 1.4901161e-08, 4.4703484e-08, 7.4505806e-09],
		 [-5.9604645e-08, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],
		
		 [[-5.7220459e-06, -8.7738037e-05, -6.0081482e-05, 1.9550323e-05],
		 [ 1.4901161e-08, 2.9802322e-08, -1.4901161e-08, -1.4901161e-08],
		 [ 0.0000000e+00, 0.0000000e+00, 5.9604645e-08, -5.9604645e-08]],
		
		 [[ 1.0108948e-04, 6.1988831e-05, -1.3351440e-04, 1.1634827e-04],
		 [ 5.2154064e-08, -1.4901161e-08, 2.9802322e-08, -2.9802322e-08],
		 [ 5.9604645e-08, -5.9604645e-08, -5.9604645e-08, 0.0000000e+00]]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_backward_affine_3():
	np.testing.assert_allclose(batchnorm_backward(4, 3, 2, 1, affine=True),
		np.array([[[[ 9.6321106e-05],
		 [ 1.4305115e-04]],
		
		 [[-7.4505806e-09],
		 [ 2.2351742e-08]],
		
		 [[ 3.5762787e-07],
		 [ 1.7881393e-07]]],
		
		
		 [[[-1.5926361e-04],
		 [-1.0013580e-04]],
		
		 [[-1.4901161e-08],
		 [ 0.0000000e+00]],
		
		 [[ 1.1920929e-07],
		 [-4.1723251e-07]]],
		
		
		 [[[ 1.4305115e-04],
		 [ 6.9618225e-05]],
		
		 [[ 0.0000000e+00],
		 [-1.4901161e-08]],
		
		 [[ 1.1920929e-07],
		 [-1.1920929e-07]]],
		
		
		 [[[ 2.0408630e-04],
		 [-3.9672852e-04]],
		
		 [[ 7.4505806e-09],
		 [ 0.0000000e+00]],
		
		 [[-1.1920929e-07],
		 [-1.7881393e-07]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_running_mean_1():
	np.testing.assert_allclose(batchnorm_running_mean(4, 3, 3),
		np.array([1.7880306, 1.6687986, 1.7533753], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_running_mean_2():
	np.testing.assert_allclose(batchnorm_running_mean(4, 3, 3, 2),
		np.array([1.5244828, 1.6152933, 1.6287339], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_running_var_1():
	np.testing.assert_allclose(batchnorm_running_var(4, 3, 2),
		np.array([1.7184839, 1.5241225, 1.9188087], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_running_var_2():
	np.testing.assert_allclose(batchnorm_running_var(4, 3, 3, 2),
		np.array([1.6113356, 1.80835 , 1.7338636], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_running_grad_1():
	np.testing.assert_allclose(batchnorm_running_grad(6, 3, 2),
		np.array([[[ 1.1005128e-07, 5.3869248e-06],
		 [-1.0843079e-05, 1.3714035e-05],
		 [ 1.7394623e-05, 8.5135298e-06]],
		
		 [[-3.2557484e-06, 2.8835300e-06],
		 [ 1.6738971e-06, 1.9555291e-05],
		 [-3.8156908e-05, 3.5037596e-05]],
		
		 [[-9.3116734e-07, 6.8174363e-06],
		 [-1.4061729e-05, 8.7072449e-06],
		 [-2.2659699e-05, -1.3659398e-05]],
		
		 [[-3.7325856e-06, -1.1660003e-05],
		 [-1.5745560e-06, 1.2641151e-05],
		 [-9.3082590e-06, 1.5248855e-05]],
		
		 [[ 8.1287390e-06, 7.8903204e-06],
		 [ 3.3130248e-06, -2.5028983e-05],
		 [ 1.7394623e-05, 6.8654619e-05]],
		
		 [[-9.1566080e-06, -2.4808880e-06],
		 [ 4.4157109e-06, -1.2512009e-05],
		 [ 4.0431819e-06, -8.2502760e-05]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_batchnorm_running_grad_2():
	np.testing.assert_allclose(batchnorm_running_grad(4, 3),
		np.array([[ 8.7022781e-06, -4.9751252e-06, 9.5367432e-05],
		 [ 6.5565109e-06, -7.2401017e-06, -2.3484230e-05],
		 [-3.5762787e-06, -4.5262277e-07, 1.6093254e-05],
		 [-1.1682510e-05, 1.2667850e-05, -8.7976456e-05]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_batchnorm():
	mugrade.submit(batchnorm_forward(2, 3))
	mugrade.submit(batchnorm_forward(7, 1, 1))
	mugrade.submit(batchnorm_forward(1, 2, 3, 4))
	mugrade.submit(batchnorm_forward(4, 3, 2, 1))
	mugrade.submit(batchnorm_forward(3, 4, affine=True))
	mugrade.submit(batchnorm_forward(3, 1, 4, affine=True))
	mugrade.submit(batchnorm_forward(3, 3, 1, 2, affine=True))
	mugrade.submit(batchnorm_forward(3, 4, 1, 3, affine=True))
	mugrade.submit(batchnorm_backward(5, 3))
	mugrade.submit(batchnorm_backward(5, 2, 4))
	mugrade.submit(batchnorm_backward(2, 1, 2, 3))
	mugrade.submit(batchnorm_backward(4, 2, 4, affine=True))
	mugrade.submit(batchnorm_running_mean(3, 1, 3))
	mugrade.submit(batchnorm_running_mean(3, 1, 3, 2))
	mugrade.submit(batchnorm_running_var(3, 1, 3))
	mugrade.submit(batchnorm_running_var(3, 1, 3, 2))
	mugrade.submit(batchnorm_running_grad(4, 3, 2))


def test_nn_dropout_forward_1():
	np.testing.assert_allclose(dropout_forward((2, 3), prob=0.45),
		np.array([[6.818182 , 0. , 0. ],
		 [0.18181819, 0. , 6.090909 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_dropout_backward_1():
	np.testing.assert_allclose(dropout_backward((2, 3), prob=0.26),
		np.array([[1.3513514, 0. , 0. ],
		 [1.3513514, 0. , 1.3513514]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_dropout():
	mugrade.submit(dropout_forward((3, 3), prob=0.4))
	mugrade.submit(dropout_backward((3, 3), prob=0.15))


def test_nn_residual_forward_1():
	np.testing.assert_allclose(residual_forward(),
		np.array([[ 2.300972 , 4.425858 , -0.30875438, 1.7591257 , 4.239749 ],
		 [-0.17286299, 3.1223633 , 2.303908 , 1.50568 , -0.38501573],
		 [-0.1970925 , 4.37333 , 2.44725 , 3.723623 , 3.313342 ],
		 [ 1.3066804 , 4.221464 , 3.2671127 , 1.7819822 , 3.8200178 ],
		 [ 0.46752548, 0.6292591 , 1.6196134 , 1.3704567 , 2.159867 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_nn_residual_backward_1():
	np.testing.assert_allclose(residual_backward(),
		np.array([[0.8737404 , 0.8007135 , 0.8190725 , 0.98652667, 1.1131225 ],
		 [0.8737404 , 0.8007135 , 0.8190725 , 0.98652667, 1.1131225 ],
		 [0.8737404 , 0.8007135 , 0.8190725 , 0.98652667, 1.1131225 ],
		 [0.8737404 , 0.8007135 , 0.8190725 , 0.98652667, 1.1131225 ],
		 [0.8737404 , 0.8007135 , 0.8190725 , 0.98652667, 1.1131225 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def submit_nn_residual():
	mugrade.submit(residual_forward(shape=(3,4)))
	mugrade.submit(residual_backward(shape=(3,4)))


def test_optim_sgd_vanilla_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.0),
		np.array(2.805675284037), rtol=1e-5, atol=1e-5)

def test_optim_sgd_momentum_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.9),
		np.array(2.7530257035), rtol=1e-5, atol=1e-5)

def test_optim_sgd_weight_decay_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.0, weight_decay=0.01),
		np.array(2.805243015835), rtol=1e-5, atol=1e-5)

def test_optim_sgd_momentum_weight_decay_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.01),
		np.array(2.754267687901), rtol=1e-5, atol=1e-5)

def test_optim_sgd_layernorm_residual_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Residual(nn.Linear(8, 8), nn.ReLU(), nn.LayerNorm(8)), nn.Linear(8, 16)), ndl.optim.SGD, epochs=3, lr=0.01, weight_decay=0.001),
		np.array(2.80843055054), rtol=1e-5, atol=1e-5)

# We're checking that you have not allocated too many tensors;
# if this fails, make sure you're using .detach()/.data whenever possible.
def test_optim_sgd_z_memory_check_1():
	np.testing.assert_allclose(global_tensor_count(),
		np.array(387), rtol=1e-5, atol=1000)

def submit_optim_sgd():
	mugrade.submit(learn_model_1d(48, 17, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 17)), ndl.optim.SGD, lr=0.03, momentum=0.0, epochs=2))
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.9, epochs=2))
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.BatchNorm(32), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.0, weight_decay=0.01, epochs=2))
	mugrade.submit(learn_model_1d(54, 16, lambda z: nn.Sequential(nn.Linear(54, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.01, epochs=2))
	mugrade.submit(learn_model_1d(64, 4, lambda z: nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Residual(nn.Linear(8, 8), nn.ReLU(), nn.LayerNorm(8)), nn.Linear(8, 4)), ndl.optim.SGD, epochs=3, lr=0.01, weight_decay=0.001))


def test_optim_adam_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, bias_correction=False),
		np.array(2.7770788338), rtol=1e-5, atol=1e-5)

def test_optim_adam_weight_decay_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.01, bias_correction=False),
		np.array(2.775645683303), rtol=1e-5, atol=1e-5)

def test_optim_adam_batchnorm_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.001, bias_correction=False),
		np.array(2.8540688, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_optim_adam_batchnorm_eval_mode_1():
	np.testing.assert_allclose(learn_model_1d_eval(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.001, bias_correction=False),
		np.array(2.8398514, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_optim_adam_layernorm_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.01, weight_decay=0.01, bias_correction=False),
		np.array(2.8333263, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_optim_adam_weight_decay_bias_correction_1():
	np.testing.assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.01, bias_correction=True),
		np.array(2.775811553566), rtol=1e-5, atol=1e-5)

# We're checking that you have not allocated too many tensors;
# if this fails, make sure you're using .detach()/.data whenever possible.
def test_optim_adam_z_memory_check_1():
	np.testing.assert_allclose(global_tensor_count(),
		np.array(1132), rtol=1e-5, atol=1000)

def submit_optim_adam():
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, bias_correction=False, epochs=2))
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.01, bias_correction=False, epochs=2))
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.BatchNorm(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.001, bias_correction=False, epochs=3))
	mugrade.submit(learn_model_1d_eval(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.BatchNorm(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.001, bias_correction=False, epochs=2))
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.01, weight_decay=0.01, bias_correction=False, epochs=2))
	mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.01, bias_correction=True, epochs=2))


def test_mlp_residual_block_num_params_1():
	np.testing.assert_allclose(residual_block_num_params(15, 2, nn.BatchNorm),
		np.array(111), rtol=1e-5, atol=1e-5)

def test_mlp_residual_block_num_params_2():
	np.testing.assert_allclose(residual_block_num_params(784, 100, nn.LayerNorm),
		np.array(159452), rtol=1e-5, atol=1e-5)

def test_mlp_residual_block_forward_1():
	np.testing.assert_allclose(residual_block_forward(15, 10, nn.LayerNorm, 0.5),
		np.array([[0. , 0.13007787, 0. , 0.6385895 , 0. ,
		 0. , 0.39603564, 0. , 0. , 0. ,
		 0.22576472, 3.068371 , 0.2753922 , 0. , 2.7325933 ]],
		 dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_mlp_resnet_num_params_1():
	np.testing.assert_allclose(mlp_resnet_num_params(150, 100, 5, 10, nn.LayerNorm),
		np.array(68360), rtol=1e-5, atol=1e-5)

def test_mlp_resnet_num_params_2():
	np.testing.assert_allclose(mlp_resnet_num_params(10, 100, 1, 100, nn.BatchNorm),
		np.array(21650), rtol=1e-5, atol=1e-5)

def test_mlp_resnet_forward_1():
	np.testing.assert_allclose(mlp_resnet_forward(10, 5, 2, 5, nn.LayerNorm, 0.5),
		np.array([[-0.5742254 , -0.7026584 , -1.0409098 , -0.110928625,
		 1.4221442 ],
		 [-0.5789626 , -0.6993287 , -1.042216 , -0.110530436,
		 1.4283872 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_mlp_resnet_forward_2():
	np.testing.assert_allclose(mlp_resnet_forward(15, 25, 5, 14, nn.BatchNorm, 0.0),
		np.array([[ 0.63092315 , -1.0447866 , -0.48108655 , 1.2441598 ,
		 -2.8625185 , -0.098286316, 0.8770414 , 1.4918139 ,
		 1.8280764 , 2.2639813 , -1.366471 , -1.8701926 ,
		 -2.513506 , -0.4411495 ],
		 [-0.57600164 , 1.6272497 , 0.24485393 , -0.8704225 ,
		 1.1589925 , 0.0757428 , -0.44717863 , 0.78446764 ,
		 1.8449777 , 0.011564424, 0.91179746 , -2.4464502 ,
		 -0.14903912 , 0.9948435 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_mlp_train_epoch_1():
	np.testing.assert_allclose(train_epoch_1(5, 250, ndl.optim.Adam, lr=0.01, weight_decay=0.1),
		np.array([0.653016666667, 1.203555985789]), rtol=0.0001, atol=0.0001)

def test_mlp_eval_epoch_1():
	np.testing.assert_allclose(eval_epoch_1(10, 150),
		np.array([0.1174 , 2.375525220633]), rtol=1e-5, atol=1e-5)

def test_mlp_train_mnist_1():
	np.testing.assert_allclose(train_mnist_1(250, 2, ndl.optim.SGD, 0.001, 0.01, 100),
		np.array([0.7274 , 1.021359381328, 0.8108 , 0.749822602421]), rtol=0.001, atol=0.001)

def submit_mlp_resnet():
	mugrade.submit(residual_block_num_params(17, 13, nn.BatchNorm))
	mugrade.submit(residual_block_num_params(785, 101, nn.LayerNorm))
	mugrade.submit(residual_block_forward(15, 5, nn.LayerNorm, 0.3))
	mugrade.submit(mlp_resnet_num_params(75, 75, 3, 3, nn.LayerNorm))
	mugrade.submit(mlp_resnet_num_params(15, 10, 10, 5, nn.BatchNorm))
	mugrade.submit(mlp_resnet_forward(12, 7, 1, 6, nn.LayerNorm, 0.8))
	mugrade.submit(mlp_resnet_forward(15, 3, 2, 15, nn.BatchNorm, 0.3))
	mugrade.submit(train_epoch_1(7, 256, ndl.optim.Adam, lr=0.01, weight_decay=0.01))
	mugrade.submit(eval_epoch_1(12, 154))
	mugrade.submit(train_mnist_1(554, 3, ndl.optim.SGD, 0.01, 0.01, 7))
