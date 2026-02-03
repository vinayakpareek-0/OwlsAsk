# Replace with your actual string and password
ATLAS_URI = "mongodb+srv://vinay:Lpp38xBlQ2hTrWiH@cluster0.tooqapc.mongodb.net/?appName=Cluster0"

# Initialize with the cloud URI
logger = AuditLogger(atlas_uri=ATLAS_URI)
session_id = str(uuid.uuid4())