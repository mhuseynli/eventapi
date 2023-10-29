import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Webshop, Event, User, Product
from .serializers import WebshopSerializer, EventSerializer

# Load the dataset
data = pd.read_csv('data.csv')

# Encode the product_id and tag values into numerical representations
label_encoders = {}


# Define the model architecture
class SessionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SessionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, _ = self.rnn(embedded)
        output = self.out(output[:, -1, :])
        return output


# Create a global variable for user_sessions
user_sessions = None


@api_view(['POST'])
def train(request):
    webshop_id = request.data.get('webshop_id')
    global user_sessions  # Make user_sessions accessible globally

    # Filter the data based on webshop_id and keep relevant columns
    filtered_data = data[data['webshop_id'] == webshop_id][['product_id', 'tag', 'user_id']]

    # Group the data by user_id to create user sessions
    user_sessions = filtered_data.groupby('user_id').apply(lambda x: x).reset_index(drop=True)

    # Encode the product_id and tag values into numerical representations
    for column in ['product_id', 'tag']:
        label_encoder = LabelEncoder()
        user_sessions[column] = label_encoder.fit_transform(user_sessions[column])
        label_encoders[column] = label_encoder

    # Rest of the training code remains the same
    train_data, val_data = train_test_split(user_sessions, test_size=0.2, random_state=42)

    input_size = len(label_encoders['product_id'].classes_)
    hidden_size = 128
    output_size = input_size
    num_layers = 1

    # Create an instance of the model
    model = SessionRNN(input_size, hidden_size, output_size, num_layers)

    # Step 6: Train the Model

    # Define the training parameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Custom dataset class for training and validation
    class RecommenderDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            row = self.data.iloc[index]
            inputs = torch.tensor(row[['product_id', 'tag']].values, dtype=torch.long)
            target = torch.tensor(row['product_id'], dtype=torch.long)
            return inputs, target

    # Create data loaders for training and validation
    train_dataset = RecommenderDataset(train_data)
    val_dataset = RecommenderDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        torch.save(model.state_dict(), 'session_rnn_model.pth')
        # Calculate the average training loss for the epoch
        average_loss = total_loss / len(train_dataset)

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Calculate the average validation loss for the epoch
        average_val_loss = val_loss / len(val_dataset)

        # Print the epoch and loss information
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}')
        return Response('Training finished')


@api_view(['POST'])
def predict(request):
    user_id = request.data.get('user_id')
    top_n = request.data.get('top_n', 5)  # Retrieve the top_n parameter from the request, default to 5 if not provided
    global user_sessions  # Access the user_sessions variable

    # Load the trained model
    model = SessionRNN(len(label_encoders['product_id'].classes_), 128, len(label_encoders['product_id'].classes_), 1)
    model.load_state_dict(torch.load('session_rnn_model.pth'))
    model.eval()

    with torch.no_grad():
        user_data = user_sessions[user_sessions['user_id'] == user_id]
        user_inputs = torch.tensor(user_data[['product_id', 'tag']].values, dtype=torch.long)
        outputs = model(user_inputs)
        _, predicted_indices = torch.topk(outputs, top_n, dim=1)
        recommended_products = []
        for indices in predicted_indices:
            products = label_encoders['product_id'].inverse_transform(indices.numpy())
            recommended_products.append(products.tolist())
    return Response({'recommendations': recommended_products})


@api_view(['POST'])
def get_users(request):
    webshop_id = request.data.get('webshop_id')
    if webshop_id:
        df = pd.read_csv('data.csv')
        filtered_df = df[df['webshop_id'] == int(webshop_id)]

        # Check if there are enough unique user IDs available
        if len(filtered_df) <= 5:
            user_data = [{'user_id': row['user_id'], 'product_id': row['product_id'], 'tag': row['tag']} for _, row in
                         filtered_df.iterrows()]
        else:
            user_data = []

            # Keep track of already returned user IDs
            returned_ids = set()

            for _, row in filtered_df.iterrows():
                user_id = row['user_id']

                # Skip if user ID is already returned
                if user_id in returned_ids:
                    continue

                user_data.append({'user_id': user_id, 'product_id': row['product_id'], 'tag': row['tag']})
                returned_ids.add(user_id)

                # Break the loop if 5 unique user IDs are collected
                if len(user_data) == 5:
                    break

        return Response({'user_data': user_data})
    else:
        return Response({'error': 'Invalid webshop_id'})


@api_view(['POST', 'GET'])
def webshop_list(request):
    if request.method == 'POST':
        serializer = WebshopSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'GET':
        webshops = Webshop.objects.all()
        serializer = WebshopSerializer(webshops, many=True)
        return Response(serializer.data)


@api_view(['GET', 'PUT', 'DELETE'])
def webshop_detail(request, pk):
    try:
        webshop = Webshop.objects.get(pk=pk)
    except Webshop.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = WebshopSerializer(webshop)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = WebshopSerializer(webshop, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        webshop.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET', 'POST'])
def event_list(request):
    if request.method == 'POST':
        webshop_id = request.data.get('webshop_id')
        user_id = request.data.get('user_id')
        product_id = request.data.get('product_id')
        event_type = request.data.get('event_type')
        tag = request.data.get('tag')

        try:
            webshop = Webshop.objects.get(pk=webshop_id)
        except Webshop.DoesNotExist:
            return Response({'error': 'Webshop does not exist.'}, status=status.HTTP_404_NOT_FOUND)

        user, _ = User.objects.get_or_create(webshop_user_id=user_id, webshop=webshop)

        product, _ = Product.objects.get_or_create(webshop_product_id=product_id, tag=tag, webshop=webshop)

        event = Event(user=user, webshop=webshop, product=product, event_type=event_type)
        event.save()

        serializer = EventSerializer(event)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    elif request.method == 'GET':
        webshop_id = request.data.get('webshop_id')
        user_id = request.data.get('user_id')

        if webshop_id and user_id:
            try:
                webshop = Webshop.objects.get(pk=webshop_id)
                user = User.objects.get(webshop_user_id=user_id, webshop=webshop)
                events = Event.objects.filter(user=user)
                serializer = EventSerializer(events, many=True)
                return Response(serializer.data)
            except (Webshop.DoesNotExist, User.DoesNotExist):
                return Response({'error': 'Webshop or User does not exist.'}, status=status.HTTP_404_NOT_FOUND)

        else:
            events = Event.objects.all()
            serializer = EventSerializer(events, many=True)
            return Response(serializer.data)
