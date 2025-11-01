import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
from torchvision.utils import make_grid
from PIL import Image
from model_gan import Generator
from torchvision import transforms 


# MNIST class labels
mnist_labels = [
        '1', '2', '3', '4', '5',
        '6', '7', '8', '9', '0'
    ]

app = FastAPI()
z_dim = 100
device=torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform=transforms.ToPILImage()
gen=Generator(z_dim)
checkpoint = torch.load("checkpoints/gan_epoch_10.pth", map_location=device)
gen.load_state_dict(checkpoint["generator_state_dict"])
gen.to(device).eval()


@app.get("/generate")
def generate(num_samples: int = 16):
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        fake = gen(z).cpu()
        fake = (fake + 1) / 2.0
        grid = make_grid(fake, nrow=int(num_samples**0.5))
        img = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil = transform(img)
  
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)

        #pil = Image.fromarray(img.squeeze(), mode="L")
        #buf = io.BytesIO()
        #pil.save(buf, format="PNG")
        #pil.save("output.jpeg")
        #buf.seek(0)
    #return Response(content=buf.getvalue(), media_type="image/png")
    #return {"status": "code success"}
    return StreamingResponse(buf, media_type="image/png")