% matplotlib
inline


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Training and Evaluation Loop
epoch_losses = []
test_losses = []
epochs = 400
for epoch in range(epochs):
    train_loss = train(train_dataloader, model, criterion, optimizer, device)
    test_loss = train(test_dataloader, model, criterion, optimizer, device)
    epoch_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Testing Loss: {test_loss}')


# Visualize the molecule with attention scores
def visualize_molecule_with_attention(mol, attention_scores):
    fig, ax = plt.subplots(figsize=(10, 10))
    norm = plt.Normalize(vmin=0, vmax=1)
    mol = Chem.Mol(mol)

    def get_color(value):
        cmap = plt.colormaps.get_cmap('coolwarm')
        return cmap(norm(value))

    atom_colors = {i: get_color(attention_scores[i]) for i in range(mol.GetNumAtoms())}

    img = SimilarityMaps.GetSimilarityMapFromWeights(mol, attention_scores, colorMap=plt.colormaps.get_cmap('coolwarm'))
    ax.imshow(img)
    ax.axis('off')
    plt.show()


# Function to convert test data to an RDKit molecule
def convert_to_mol(data):
    atom_features = data.x
    bond_features = data.edge_attr
    edge_index = data.edge_index

    mol = Chem.RWMol()
    atom_map = {}
    for i in range(atom_features.shape[0]):
        atom = Chem.Atom(int(atom_features[i][0]))
        idx = mol.AddAtom(atom)
        atom_map[i] = idx

    added_bonds = set()
    for i in range(edge_index.shape[1]):
        start, end = edge_index[:, i]
        start_idx = atom_map[int(start)]
        end_idx = atom_map[int(end)]
        bond = (min(start_idx, end_idx), max(start_idx, end_idx))
        if bond not in added_bonds:
            bond_type = Chem.rdchem.BondType.SINGLE  # Adjust as necessary
            mol.AddBond(start_idx, end_idx, bond_type)
            added_bonds.add(bond)

    return mol


# Visualizing the second molecule in the dataset
second_data = dataset[203]  # Selecting the second molecule
model.eval()
with torch.no_grad():
    output, att_scores1, att_scores2, att_scores3 = model(second_data.to(device))
    att_score_sum = att_scores1 + att_scores2 + att_scores3

# Normalize attention scores for visualization
att_score_sum = (att_score_sum - att_score_sum.min()) / (att_score_sum.max() - att_score_sum.min())

# Convert to RDKit molecule and visualize
second_mol = convert_to_mol(second_data)
visualize_molecule_with_attention(second_mol, att_score_sum.cpu().numpy())
