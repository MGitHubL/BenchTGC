import time

from library_data import *
import library_models as lib
from library_models import *
from evaluation import eva
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.autograd import Variable

FType = torch.FloatTensor
LType = torch.LongTensor

data = 'school'
k_dict = {'arxivAI': 5, 'arxivCS': 40, 'arxivPhy': 53, 'arxivMath': 31, 'arxivLarge': 172, 'school': 9, 'dblp': 10,
          'brain': 10, 'patent': 6}
parser = argparse.ArgumentParser()
parser.add_argument('--network', default=data, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--clusters', default=k_dict[data], help='Model name to save output in file')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
args = parser.parse_args()

args.datapath = "../../data/%s/%s.txt" % (args.network, args.network)
args.pre_emb_path = '../pretrain/%s_feature.emb' % (args.network)

pre_node_emb = dict()
with open(args.pre_emb_path, 'r') as reader:
    reader.readline()
    for line in reader:
        embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
        node_id = embeds[0]
        pre_node_emb[node_id] = embeds[1:]
    reader.close()
feature = []
for i in range(len(pre_node_emb)):
    feature.append(pre_node_emb[i])
pre_feature = np.array(feature)
pre_emb = Variable(torch.from_numpy(pre_feature).type(FType).cuda(), requires_grad=False)

# SET GPU
args.gpu = str(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, y_true, node_set] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1  # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true) / (1.0 + sum(y_true))

timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 10000

# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1, true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

node_embeddings = nn.Parameter(pre_emb)

model.cluster_layer = (torch.zeros(args.clusters, args.embedding_dim) + 1.).cuda()
torch.nn.init.xavier_normal_(model.cluster_layer.data).cuda()

kmeans = KMeans(n_clusters=args.clusters, n_init=20)
_ = kmeans.fit_predict(pre_feature)
model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
pre_cluster = model.cluster_layer.data.clone()
v = 1.0

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}


for ep in range(args.epochs):

    epoch_start_time = time.time()

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count = 0, 0, 0

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False

    for j in range(num_interactions):

        if is_first_epoch:
            # READ INTERACTION J
            userid = user_sequence_id[j]
            itemid = item_sequence_id[j]
            feature = feature_sequence[j]
            user_timediff = user_timediffs_sequence[j]
            item_timediff = item_timediffs_sequence[j]

            tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
            lib.tbatchid_user[userid] = tbatch_to_insert
            lib.tbatchid_item[itemid] = tbatch_to_insert

            lib.current_tbatches_user[tbatch_to_insert].append(userid)
            lib.current_tbatches_item[tbatch_to_insert].append(itemid)
            lib.current_tbatches_feature[tbatch_to_insert].append(feature)
            lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
            lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
            lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
            lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

        timestamp = timestamp_sequence[j]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # ITERATE OVER ALL T-BATCHES
            if not is_first_epoch:
                lib.current_tbatches_user = cached_tbatches_user[timestamp]
                lib.current_tbatches_item = cached_tbatches_item[timestamp]
                lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

            for i in range(len(lib.current_tbatches_user)):
                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                # LOAD THE CURRENT TBATCH
                if is_first_epoch:
                    lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                    lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                    lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                    lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()

                    lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                    lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                    lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()

                tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_interactionids = lib.current_tbatches_interactionids[i]
                feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                item_embedding_previous = node_embeddings[tbatch_itemids_previous,:]

                # PROJECT USER EMBEDDING TO CURRENT TIME
                user_embedding_input = node_embeddings[tbatch_userids,:]
                user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous], dim=1)

                # PREDICT NEXT ITEM EMBEDDING
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                # CALCULATE PREDICTION LOSS
                item_embedding_input = node_embeddings[tbatch_itemids,:]
                loss += MSELoss(predicted_item_embedding, item_embedding_input.detach())

                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                user_pre_emb = pre_emb[tbatch_userids, :]
                item_pre_emb = pre_emb[tbatch_itemids, :]
                l_x = torch.norm(user_embedding_output - user_pre_emb, p=2) + torch.norm(
                    item_embedding_output - item_pre_emb, p=2) + 1e-6  # []

                l_framework = l_x

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                loss += l_framework

                node_embeddings[tbatch_itemids,:].data = item_embedding_output.data.clone()
                node_embeddings[tbatch_userids,:].data = user_embedding_output.data.clone()

                pre_cluster.data = model.cluster_layer.data.clone()

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = 0
            if is_first_epoch:
                cached_tbatches_user[timestamp] = lib.current_tbatches_user
                cached_tbatches_item[timestamp] = lib.current_tbatches_item
                cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item

                reinitialize_tbatches()
                tbatch_to_insert = -1

    is_first_epoch = False  # as first epoch ends here
    # END OF ONE EPOCH
    path = '../../emb/%s/%s_IJODIE.emb' % (args.network, args.network)

    embeddings = node_embeddings.cpu().data.numpy()
    writer = open(path, 'w')
    writer.write('%d %d\n' % (len(node_set), args.embedding_dim))
    for n_idx in range(len(node_set)):
        writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

    writer.close()

    sys.stdout.write("\rTotal loss in epoch %d: %f  " % (ep, total_loss))
    sys.stdout.flush()

print("*** Training complete. ***")


