import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import glob, xmltodict

if __name__ == "__main__":
	folder = "../hievents/"
	files_list = open("files_list", "w")
	for file in glob.glob(folder + "*.xml"):
		fd = open(file)
		doc = xmltodict.parse(fd.read())
		
		text = doc["ArticleInfo"]["Text"]

		output_file = file.split("/")[-1].replace(".xml", ".txt")
		files_list.write("text/"+output_file+"\n")
		output = open(output_file, "w")
		output.write(text)
		output.close()
	files_list.close()
		