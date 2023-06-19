import datetime
import PIL
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection



def generate_info_licences():
  """
    Generates dictionatry for information and licenses.

    This function creates and returns a dictionary containing information about the licenses,
    as well as a list of licenses. The information includes the current year, version number,
    description, contributor, URL, and date of creation.

    Returns:
        tuple: A tuple containing two elements:
            - info (dict): A dictionary with the following keys:
                - 'year' (str): The current year.
                - 'version' (int): The version number (currently set to 1).
                - 'description' (str): An empty string for the description.
                - 'contributor' (str): The name of the contributor, set to 'Redwood twins'.
                - 'url' (str): An empty string for the URL.
                - 'date_created' (str): The date and time of creation.
            - licences (list): A list of dictionaries, each representing a license, with the following keys:
                - 'id' (int): The license ID.
                - 'url' (str): An empty string for the URL.
                - 'name' (str): The name of the license, set to 'RWT'.
    """

  info = {'year': str(datetime.date.today().year),
        'version': 1,
        'description': '',
        'contributor': 'Redwood twins',
        'url' : '',
        'date_created': str(datetime.datetime.now())}

  licences = [{'id': 1, 'url': '', 'name': 'RWT'}]
  return info,licences

def generate_empty_category():
  """
    Generates an empty category.

    This function creates and returns a dictionary representing an empty category. 
    The dictionary has the following keys:
        - 'id' (int): The ID of the category, set to 0.
        - 'name' (str): The name of the category, set to an empty string.
        - 'supercategory' (str): The supercategory of the category, set to an empty string.

    Returns:
        dict: A dictionary representing an empty category with the keys mentioned above.
    """
  category = {
      'id': 0, 
      'name': '', 
      'supercategory': ''
    }
  
  return category

def generate_empty_annotation():
  """
    Generates an empty annotation.

    This function creates and returns a dictionary representing an empty annotation.
    The dictionary has the following keys:
        - 'id' (int): The ID of the annotation, set to 0.
        - 'image_id' (int): The ID of the associated image, set to 0.
        - 'category_id' (int): The ID of the associated category, set to 0.
        - 'bbox' (list): An empty list representing the bounding box coordinates.
        - 'area' (int): The area of the annotation, set to 0.
        - 'segmentation' (list): An empty list representing the segmentation data.
        - 'iscrowd' (int): A flag indicating whether the annotation represents a crowd, set to 0.

    Returns:
        dict: A dictionary representing an empty annotation with the keys mentioned above.
    """
  bbox = []
  annotation = {
    'id': 0,
    'image_id': 0,
    'category_id': 0,
    'bbox': bbox,
    'area': 0,
    'segmentation': [],
    'iscrowd': 0}
  
  return annotation

def generate_empty_image():
  """
    Generates an empty image.

    This function creates and returns a dictionary representing an empty image.
    The dictionary has the following keys:
        - 'id' (int): The ID of the image, set to 0.
        - 'licence' (str): The license associated with the image, set to 'RWT'.
        - 'file_name' (str): The name of the image file, set to an empty string.
        - 'height' (int): The height of the image, set to 0.
        - 'width' (int): The width of the image, set to 0.
        - 'date_captured' (str): The date and time when the image was captured, represented as a string.

    Returns:
        dict: A dictionary representing an empty image with the keys mentioned above.
    """
  image = {
    'id': 0,
    'licence': 'RWT',
    'file_name': '',
    'height': 0,
    'width': 0,
    'date_captured': str(datetime.datetime.now())
  }
  return image

def generate_bounding_box(base_path_val,val_images ):

  """
    Generates bounding box coordinates for object detection.

    This function takes an image file, performs object detection using a pre-trained model,
    and returns the bounding box coordinates of the detected objects. The function utilizes
    a preprocessor, object detection model, and a base path to locate the image. It opens
    the image using the PIL library, processes it with the preprocessor, runs the object
    detection model, and extracts the bounding box coordinates, labels, and scores for the
    detected objects. The results are stored in a dictionary, where the image filename is used
    as the key, and the bounding box information is stored as a value.

    Args:
        base_path_val (str): The base path where the image is located.
        val_images (list str): List of image names to be processed.

    Returns:
        dict: A dictionary containing the file name, scores, labels and bounding box coordinates 
        for the detected objects.
    """
  resultados=[ ]
  
  #
  processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
  #base_path_val = 
  #for file  in base_path_val:
  for name in val_images:
    dict_result = {}

  

    oneimage = PIL.Image.open(base_path_val+name)

    inputs = processor(images=oneimage, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([oneimage.size[::-1]])
    result = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    result['scores'] = result['scores'].tolist()
    result['labels'] = result['labels'].tolist()
    result['boxes'] = result['boxes'].tolist()
    dict_result[name] = result
    resultados.append(dict_result)
  return jsonGenerator(base_path_val,resultados, model)


def jsonGenerator(*args):
  base_path = args[0]
  results_val = args[1]
  model = args[2]
  info,licences=generate_info_licences()
  categories =[]
  category = generate_empty_category()
  images =[]
  annotations = []
  bbox=[]
  
  annotations_coco = {
    'info': info,
    'licences':licences,
    'categories':categories,
    'images': images,
     'annotations': annotations

}
  
  for result in results_val:
    image = generate_empty_image()
    for key, value in result.items():
      image['file_name'] = key
      image['id']=len(images)
    
      im = Image.open(base_path+key) #ruta definir en parte superior y solo llamar    
      width, height = im.size
      image['width']= width
      image['height']= height
      images.append(image)
      for score,  label, box in zip(value['scores'],value['labels'], value['boxes']):
        annotation = generate_empty_annotation()
        category = generate_empty_category()
        annotation['id'] =len(annotations)
        category['name'] = model.config.id2label[label]
        if model.config.id2label[label] not in [x['name'] for x in categories]:
          category['id'] = len(categories)
          annotation['category_id'] = len(categories)
          categories.append(category)


        bbox = [round(i, 2) for i in box]
        annotation['area'] = bbox[2]*bbox[3]
        annotation['bbox'] = bbox
        annotation['image_id'] = len(images)
      
        annotations.append(annotation)
  return annotations_coco
