import torch
from model import ClassificationModel
from torchvision import transforms
from config import classes
from PIL import  Image

modelpath = 'epoch=9-step=1769.ckpt'
model = ClassificationModel()
model.load_state_dict(torch.load(modelpath)['state_dict'], strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


transform = transforms.Compose([
                transforms.ToTensor(),
])


def check_for_gloves(image):
    with torch.no_grad():
        image = transform(image)
        print(image.shape)
        image = image.unsqueeze(0)
        print(image.shape)
        image = image.to(device)
        print(image.shape)
        output = model(image)

    op, predicted = torch.max(output.data, 1)
    print(predicted.item(),  classes[predicted.item()], op.item())
    return predicted.item(),  classes[predicted.item()], op.item()


# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     check_for_gloves(frame)
#
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


image = Image.open('')
check_for_gloves(image)
