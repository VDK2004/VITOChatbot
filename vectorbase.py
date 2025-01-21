from pinecone import Pinecone, ServerlessSpec

# API-sleutel en regio instellen
PINECONE_API_KEY = "pcsk_4v8YTF_6Sgtnwh2Tm38koMurffJJLUp4eyHncT843KmkeKN3GVbjqSNbFPtzjBiDTfkF6V"
INDEX_NAME = "projectvito"  # Pas dit aan naar jouw indexnaam

# Maak een Pinecone-instantie
pc = Pinecone(
    api_key=PINECONE_API_KEY,
)

# Controleer of de index al bestaat, anders maak je deze aan
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Pas dit aan naar de dimensie van jouw embeddings
        metric='cosine',  # Gebruik 'cosine', 'euclidean', of 'dotproduct' naar wens
        spec=ServerlessSpec(
            cloud='aws',  # Specificeer de cloudprovider (bv. 'aws', 'gcp', etc.)
            region='us-east-1'  # Gebruik de gewenste regio (hier 'us-east-1')
        )
    )

# Verbind met de bestaande index
index = pc.Index(INDEX_NAME)

# Test of alles correct werkt
print(f"Verbonden met index: {INDEX_NAME}")
