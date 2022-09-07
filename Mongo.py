from pymongo import MongoClient
import base64
import gridfs
from io import BytesIO
from PIL import Image

cluster = MongoClient("MongoToken")

dbp = cluster["Sign_Translate"]
collections = dbp["Fotico"]

fs = gridfs.GridFS(dbp)

async def GuardarUsuario(user,image,labels):
    encoded_string = base64.b64encode(image)
    image_id = fs.put(encoded_string)
    if collections.count_documents({ "_id": user }) != 0:
        collections.update_one({ "_id": user }, {'$push': {'galeria': {"imagen":image_id,"label_correcta": labels[0],"label_probables":labels[1:]}}})
    else:
        post = {"_id": user,"galeria":[{"imagen":image_id, "label_correcta":labels[0],"label_probables":labels[1:]}]}
        collections.insert_one(post)

async def GetImagenes(user):
    if collections.count_documents({ "_id": user }) != 0:
        myquery = { "_id": user }
        result = collections.find_one(myquery)
        fotos = {fs.get(foto['imagen']):foto['label_correcta'] for foto in result['galeria']}
        datos = [(Image.open(BytesIO(base64.b64decode(img.read()))),label) for img,label in fotos.items()]
        return datos
    return None

async def CerrarDB(ctx):
    cluster.close()