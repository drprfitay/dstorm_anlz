from flask import Flask, redirect, url_for, request, render_template, jsonify
from os import listdir, path, walk, mkdir, unlink, environ
from shutil import rmtree
from scan_thread import scanThread
import json
import threading

app = Flask(__name__) 
app.config['TEMPLATES_AUTO_RELOAD'] = True
UPLOAD_FOLDER = 'experimentfiles/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dirs = [d[0] for d in walk("experimentfiles/")]
experimentfiles_dict = ([(d, listdir(d)) for d in dirs])
experimentfiles_dict = dict([(d, [e for e in f if ".csv" in e]) for d,f in experimentfiles_dict])


num_of_files_in_page = 25

scan_results = []
scan_thread = None
prescan_thread = None


def refresh_folders():
	global dirs
	global experimentfiles_dict
	dirs = [d[0] for d in walk("experimentfiles/")]
	experimentfiles_dict = ([(d, listdir(d)) for d in dirs])
	experimentfiles_dict = dict([(d, [e for e in f if ".csv" in e]) for d,f in experimentfiles_dict])

@app.route('/exfiles', methods=['POST', 'GET'])
def exfiles():
	global experimentfiles_dict

	pagenum = request.args.get("pagenum")
	searchkey = request.args.get("searchkey")
	folder = request.args.get("folder")
	folder = "experimentfiles/" if folder is None else folder
	modified_experimentfiles = experimentfiles_dict[folder]
	requested_page = None
	
	if searchkey is not None:
		modified_experimentfiles = [f for f in experimentfiles_dict[folder] if searchkey in f]

	total_files = len(modified_experimentfiles)
	if pagenum is not None:
		pagenum = int(pagenum)
		if pagenum < 1:
			pagenum = 1
		requested_page = pagenum
		modified_experimentfiles =\
			modified_experimentfiles[(pagenum - 1) * num_of_files_in_page : pagenum * num_of_files_in_page]

	response_json = {"exfiles" : modified_experimentfiles, "total_files" :total_files,\
	 				 "total_file_in_response": len(modified_experimentfiles), "requested_page": requested_page,\
	 				 "search_key" : searchkey, "items_in_page": num_of_files_in_page, "folder": folder}	

	return jsonify(response_json)

@app.route('/results',methods = ['POST', 'GET']) 
def resultpage():
	return render_template("scanresults.html")

@app.route('/resultsforprescan',methods = ['POST', 'GET']) 
def resultsforprescan():
	return render_template("prescanresults.html")

@app.route('/folders', methods=['POST', 'GET'])
def folders():
	return jsonify({"folders": dirs})

@app.route('/uploadfiles', methods=['POST', 'GET'])
def uploadfiles():
	res = 0
	try:	
		workingfolder = request.form["current_folder"];
		files = request.files.getlist("file[]")
		for file in files:

			full_path = path.abspath("./" + workingfolder + "/" +  file.filename)
			print("Attempting to upload %s" % full_path)

			if ".csv" not in file.filename:
				print("Not a csv file")
				continue

			file.save(full_path)
			print("Uploaded %s" % full_path)
			res += 1

		refresh_folders()	
	except BaseException as be:
		print(be)

	return jsonify({"res": res})


@app.route('/createfolders', methods=['POST', 'GET'])
def createfolders():
	res = False
	try:	
		new_folder_name = request.form["new_folder"]
		folder = request.form["current_folder"]
		new_path = path.abspath("./" + folder + "/" + new_folder_name)
		mkdir(new_path)
		refresh_folders()
		print("Creating new folder: %s" % new_path)
		res = True
	except BaseException as be:
		print(be)

	return jsonify({"res": res})

@app.route('/deletefolder', methods=['POST', 'GET'])
def deletefolder():
	res = False
	try:	
		folder = request.form["deleted_folder"]
		del_path = path.abspath("./" + folder )
		rmtree(del_path)
		refresh_folders()
		print("Delete folder: %s" % del_path)
		res = True
	except BaseException as be:
		print(be)

	return jsonify({"res": res})

@app.route('/deletefiles', methods=['POST', 'GET'])
def deletefiles():
	res = False
	try:	
		post_req_files = request.form["files"]
		if (post_req_files is not None):
			files = list(set(post_req_files.split(";")[:-1]))
			full_path_files = [path.abspath("./" + f) for f in files]

			for f in full_path_files:
				unlink(f)
				print("Deleting file: %s" % f)

			refresh_folders()
			res = True
	except BaseException as be:
		print(be)

	return jsonify({"res": res})


@app.route('/',methods = ['POST', 'GET']) 
def index():
	return render_template("index.html")

@app.route('/upload_files',methods = ['POST', 'GET']) 
def upload_files():
	return render_template("upload_files.html")	

@app.route('/runprescan',methods = ['POST', 'GET']) 
def runprescan():
	return render_template("runprescan.html")

@app.route('/runscan',methods = ['POST', 'GET']) 
def runscan():
	return render_template("runscan.html", 
						   files=experimentfiles_dict["experimentfiles/"],
						   files_num=len(experimentfiles_dict["experimentfiles/"]))

@app.route('/doscan', methods = ['POST', 'GET'])
def doscan():
	print("Attempting to preform a new scan!")
	
	global scan_thread
	post_req_files = request.form["files"]
	gen_threeD = True if request.form["tdchecked"] == 'true' else False

	user_multiple_configs = []
	# To me, THIS IS NOT A GOOD EXAMPLE OF HOW TO WRITE CODE!

	num_of_confs = len(request.form["use_hdbscan"].split(";")) - 1
	print(request.form["use_hdbscan"])
	for i in range(0, num_of_confs):
		user_config = {}
		user_config["dbscan"] = {}
		user_config["hdbscan"] = {}
		user_config["coloc"] = {}
		user_config["k-dist"] = {}
		user_config["general"] = {}
		user_config["dbscan"]["dbscan_eps"] = 			   int(request.form["dbscan_dbscan_eps"].split(";")[i])
		user_config["dbscan"]["dbscan_min_samples"] =      int(request.form["dbscan_dbscan_min_samples"].split(";")[i])
		user_config["dbscan"]["min_npoints"] = 			   int(request.form["dbscan_min_npoints"].split(";")[i])
		user_config["coloc"]["coloc_neighbors"] = 		   int(request.form["coloc_coloc_neighbors"].split(";")[i])
		user_config["coloc"]["coloc_distance"] = 		   int(request.form["coloc_coloc_distance"].split(";")[i])
		user_config["coloc"]["workers"] = 				   int(request.form["coloc_workers"].split(";")[i])
		user_config["general"]["noise_reduce"] = 		   True if request.form["noise_reduce"].split(";")[i] == "true" else False
		user_config["general"]["use_z"] = 				   True if request.form["use_z"].split(";")[i] == "true" else False
		user_config["general"]["use_hdbscan"] = 		   True if request.form["use_hdbscan"].split(";")[i] == "true" else False
		user_config["general"]["stddev_num"] = 			   float(request.form["stddev_num"].split(";")[i])	
		user_config["hdbscan"]["hdbscan_min_npoints"] =    int(request.form["hdbscan_min_npoints"].split(";")[i])
		user_config["hdbscan"]["hdbscan_min_samples"] =    int(request.form["hdbscan_min_samples"].split(";")[i])
		user_config["hdbscan"]["hdbscan_eps"] = 		   int(request.form["hdbscan_epsilon"].split(";")[i])
		if user_config["hdbscan"]["hdbscan_eps"] < 0:
			user_config["hdbscan"]["hdbscan_eps"] = -9999
		user_config["hdbscan"]["hdbscan_extracting_alg"] = str(request.form["hdbscan_extracting_alg"].split(";")[i])
		if user_config["hdbscan"]["hdbscan_extracting_alg"] not in ["leaf", "eom"]:
			user_config["hdbscan"]["hdbscan_extracting_alg"] = "leaf"
		user_config["hdbscan"]["hdbscan_alpha"] = 		   float(request.form["hdbscan_alpha"].split(";")[i])
		user_config["general"]["density_drop_threshold"] = float(request.form["density_drop_threshold"].split(";")[i])
		user_config["k-dist"]["is_3D"] =				   scanThread.default_configuration["k-dist"]["is_3D"]
		user_config["k-dist"]["k"] = 					   scanThread.default_configuration["k-dist"]["k"]
		user_config["k-dist"]["n_neighbors"] = 			   scanThread.default_configuration["k-dist"]["n_neighbors"]
		user_config["k-dist"]["algorithm"] = 			   scanThread.default_configuration["k-dist"]["algorithm"]
		user_multiple_configs.append(user_config)

	print(user_multiple_configs)

	user_theme = request.form["theme"].split(";")[0]
	if user_theme == "redo":
		if scan_thread is not None:
			user_theme = scan_thread.theme_key
		else:
			user_theme = "1"

	files = []

	if (post_req_files is not None):
		files = list(set(post_req_files.split(";")[:-1]))
	full_path_files = [path.abspath("./" + f) for f in files]

	print("####### Running new scan! #########\n")
	print("#######\n")
	print("####### %s\n" % " ".join(full_path_files))

	scan_thread = scanThread()
	scan_thread.run_scan(experiment_files=full_path_files,\
						 configuration=user_multiple_configs, 
						 gen_threeD=gen_threeD,
						 theme=user_theme)

	return ('', 204) # empty response

@app.route("/scanresults")
def scanresults():
	if scan_thread is None or scan_thread.status != scanThread.scanStatus.SCAN_FINISHED_SUCCESSFULLY:
		return jsonify({"scan_completed_successfully" : False})

	res_json = scan_thread.result_json
	res_json["scan_completed_successfully"] = True

	return jsonify(res_json)

@app.route("/prescanresults")
def prescanresults():
	global prescan_thread
	if prescan_thread is None or prescan_thread.status != scanThread.scanStatus.SCAN_FINISHED_SUCCESSFULLY:
		return jsonify({"scan_completed_successfully" : False})

	res_json = prescan_thread.result_json
	res_json["scan_completed_successfully"] = True

	return jsonify(res_json)

@app.route ('/doprescan', methods = ['POST', 'GET'])
def prescan():
	print("Attempting to preform a new prescan!")
	
	global prescan_thread
	post_req_files = request.form["files"]
	k = request.form["k"].split(";")[:-1]
	eps = request.form["epsdist"].split(";")[:-1]
	k = list(set([int(kv) for kv in k]))
	eps = list(set([int(ev) for ev in eps]))

	print(k, eps)
	files = []

	if (post_req_files is not None):
		files = list(set(post_req_files.split(";")[:-1]))

	full_path_files = [path.abspath("./" + f) for f in files]

	print("####### Running new pre scan! #########\n")
	print("#######\n")
	print("####### %s\n" % " ".join(full_path_files))

	prescan_thread = scanThread(is_prescan=True)
	prescan_thread.run_prescan(experiment_files=full_path_files, knn=k, epsilons=eps)
	return ('', 204)

@app.route('/pollscan')
def pollscan():
	global scan_thread

	if scan_thread is None:
		res = scanThread.scanStatus.SCAN_NOT_STARTED
		pctg = 0
	else:
		res = scan_thread.status
		pctg = int(scan_thread.scan_progress_percentage * 100)

	return jsonify({"status": res.value,\
					"percentage": pctg})

@app.route('/pollprescan')
def pollprescan():
	global prescan_thread

	if prescan_thread is None:
		res = scanThread.scanStatus.SCAN_NOT_STARTED
		pctg = 0
	else:
		res = prescan_thread.status
		pctg = int(prescan_thread.scan_progress_percentage * 100)

	return jsonify({"status": res.value,\
					"percentage": pctg})


@app.route('/getdefaultscanparams')
def defaultscanparams():
	return jsonify(scanThread.default_configuration)

@app.route('/getdefaultprescanparams')
def defaultprescanparams():
	return jsonify(scanThread.default_prescan)

@app.route('/lastscanparams')
def lastscanparams():
	global scan_thread

	if scan_thread is None:
		return {"scan_run": False}
	

	res = scan_thread.conf[0]
	res["scan_run"] = True
	return jsonify(res)

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=int(environ.get("PORT", 5000)), debug = False, use_reloader=False) 