from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preProcess import SimplePreprocessor
from preProcess import SimpleDatasetLoader
from imutils import paths
import argparse

#command line arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True,help="path")
ap.add_argument("-k","--neighbors",type = int,default = 1,help = "# k ")
ap.add_argument("-j","--jobs",type = int , default = -1,help = "# jobs")
args = vars(ap.parse_args())
 
print("[INFO] loading images...")
#loading images
imagePaths = list(paths.list_images(args["dataset"]))
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.reshape((data.shape[0], 3072))
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

#encoding and spliting dataset
le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)


print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])

model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX),target_names=le.classes_))

# /dataset_name/class/image.jpg

