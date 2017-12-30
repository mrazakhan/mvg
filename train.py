from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from layers.graph import GraphConvolution
from utils import *
import scipy
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import time

# Define parameters
DATASET = 'cora'
#DATASET = 'citeseer'
#DATASET='Pubmed-Diabetes'
FILTER = 'chebyshev'#'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

layer2=True
grossman=False

# Get data
if not layer2:
    X, A, y = load_data(dataset=DATASET)
else:
    X, A, y, A2 = load_data(dataset=DATASET, layer2=True)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# Normalize X
X = np.diag(1./np.array(X.sum(1)).flatten()).dot(X)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    A=preprocess_adj(A,symmetric = SYM_NORM)
    L = normalized_laplacian(A, SYM_NORM)
    print('Class {0}'.format(type(L)))
    if layer2 and grossman:
        # calculate modified laplacian
        A2=preprocess_adj(A2,symmetric = SYM_NORM)
    
        #L2,_=scipy.sparse.csgraph.laplacian(A2, normed=True)
        L2=normalized_laplacian(A2,SYM_NORM)
        print('Calculating eigen vector for first layer')
        eval1, evec1=eigsh(L)
        print('Calculating eigen vector for second layer')
        eval2, evec2=eigsh(L2)
        Lmod = np.zeros((L2.shape[0],L2.shape[0]))
        print('Adding the eigenvector contribution of first layer')
        Lmod+=evec1.dot(evec1.T)
        print('Adding the eigenvector contribution of second layer')
        Lmod+=evec2.dot(evec2.T)
        print('Final')
        L=sp.csr_matrix(L+L2-0.5*Lmod)
        print (L)
        
        print('Class {}'.format(type(L)))
    elif layer2:
        A2=preprocess_adj(A2,symmetric = SYM_NORM)
        L=normalized_laplacian(A, symmetric=False)
        L2=normalized_laplacian(A2, symmetric=True)
        eval1, evec= eigsh(L)
        Lmod=np.zeros((L.shape[0], L.shape[1]))
        for i in range(1,6):
            evec_mod=0.04*np.linalg.inv(L2+0.04*np.eye(L2.shape[0],L2.shape[1]))*evec[:,i].reshape((-1, 1))
            L[:,i]=evec_mod
            print('L shape', L.shape)
            print('evec_mod.shape', evec_mod.shape)
        #L=sp.csr_matrix(Lmod, dtype=np.float32)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16, support, activation='relu', W_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
