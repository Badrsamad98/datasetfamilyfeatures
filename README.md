This is a project that I have done with the Laboratory of Machine Intelligence, University of Ljubljana. The goal was to create a clean dataset composed of the images of 100 families with the relations father, mother and daughter. This dataset was used to predict a child's appearance based on images of the parents.

The first step was to download the images from the web using a crawler.

The crawler (web_crawler.py) searches for images of the father, mother and daughter and downloads them in a specific folder structure created at run time, and appends to each image a csv identification file (filename, name of the person, gender, kin relation, link of the picture and the names of the other members of the family)

Two text files allow the configuration of the crawler:

- input.txt: this file contains the names of each family member in the following order: father, mother then daughter. Before each family, a marker “begin” allows the separation of the families. 
begin
Father name
Mother name
Daughter name

- config.txt: contains the minimum values of the height and width of the image and the number of pictures to be retrieved.

Three scripts can be launched:
web_crawler.py : retrieves the images.
generate_landmarks.py: generates the face landmarks as a json file next to the original image. The landmarks.json file must be verified manually (an approximate 15% of the dataset has detection errors) using helper pictures, copied in the “landmarks” folder, that contain the five detected landmarks of the face features (eyes, nose and mouth) as colored points.
crop_images_from_landmarks.py: image cropping based on the corrected landmarks. This script uses the corrected json landmarks files and it does the processing, the alignment and the cropping of each image.
