from module.detector import Detector
import torch
import utils.tools as tools
import utils.image as image
import cv2

class FastestDet():
    """
    FastestDet module
    """
    def __init__(self,
                 category_length = None,
                 model_path      = None, 
                 names_path      = None,
                 device : str = "cpu"
                ):
        
        self.LABEL_NAMES = []
        with open(names_path, 'r') as f:
            for line in f.readlines():
                self.LABEL_NAMES.append(line.strip())
                                
        model_type = tools.get_extension(model_path)
        if model_type == ".pth" :
            self.model = self.load_torch_model(
                                                category_length,
                                                model_path,
                                                device
                                            )

    def load_torch_model(self, category_length, model_path, device):
        """
        load pytorch model
        """
        model = Detector(category_num =category_length, load_param = True).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def predict(self, image):
        return self.model(image)
    
    def postprocess(self, original_image, preds, 
                    thresh, device="cpu"):
        
        output = tools.handle_preds(
                            preds  = preds,
                            device = device,
                            conf_thresh = thresh
                            )
        
        H, W, _ = original_image.shape

        for box in output[0]:
            box = box.tolist()
        
            obj_score = box[4]
            category = self.LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)

            cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(original_image, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
            cv2.putText(original_image, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        return original_image



if __name__ == "__main__":

    yamlFile = "configs/fastestdet/fastestdet.yaml"
    config = tools.YamlParser(config_file = yamlFile)

    model = FastestDet(
        category_length=config.classes, # coco dataset
        model_path = config.weights,
        names_path = config.names
    )

    print("Model loaded ", model)

    # # load image
    preprocessed, original_image = image.load_test_image(
        filename    = "./data/1.jpg",
        model_type  = "fastestdet",
        width       = config.input_size.width,
        height      = config.input_size.height
    )

    results = model.predict(preprocessed)
    output  = model.postprocess(
        original_image=original_image,
        preds       = results,
        thresh      = 0.5
    )

    cv2.imwrite("results.jpg", output)

    


    
