
# ================= IMPORTS =================

import os
import sys
import hashlib
import datetime
import base64
import sqlite3
import shutil

import cv2
import pdfkit
import numpy as np
from markupsafe import escape
from flask import (
    Flask,
    session,
    render_template,
    request,
    redirect,
    url_for,
    make_response,
    jsonify,
)

# Project modules
sys.path.insert(0, "./database/")
from database import dbfunctions1 as db

sys.path.insert(0, "./preprocess/")
from preprocess import prepnew as pre
from preprocess import model as m
from preprocess import hybrid_model as hm   # <-- NEW

# ================= CONFIGURATION =================

app = Flask(__name__, static_url_path="/static")
app.secret_key = "S3cr3t_K3Y_0v3rF1ow=3D"

# ================= CONFIGURATION =================

app = Flask(__name__, static_url_path="/static")
app.secret_key = "S3cr3t_K3Y_0v3rF1ow=3D"

timeStamp = str(datetime.date.today()).replace(":", "-")
TRAIN_OUTPUT_FOLDERNAME = "E:/Project25/J_Blosmia1/static/training/retrainedimages"
DATABASE_PATH = "./database/database1.db"

# ================= AUTHENTICATION ROUTES =================


# Login page for admin authentication."
@app.route("/", methods=["GET", "POST"])
def index():  # change this index, write without json
    if request.method == "GET":
        return render_template("index.html")
    else:
        if request.is_json:
            data = request.get_json()
            username = data.get("username")
            password = data.get("password")
            if not username or not password:
                return jsonify({"message": "Username and password are required"}), 400
            try:
                auth = db.admin(
                    username, password
                )  # Assuming db.admin checks user credentials
            except Exception as e:
                return jsonify({"message": f"Error occurred: {str(e)}"}), 500

            if len(auth) > 0:
                session["user"] = username  # Store username in session
                return jsonify({"message": "Login successful", "user": username}), 200
            else:
                return (
                    jsonify(
                        {
                            "message": "Incorrect credentials. Please enter the correct credentials"
                        }
                    ),
                    401,
                )
        else:
            return jsonify({"message": "Content-Type must be application/json"}), 415


# Logout and clear session.
@app.route("/dropsession")
def dropsession():
    session.pop("user", None)
    session.pop("admin", None)
    return redirect(url_for("index"))


# Dashboard page.
@app.route("/dash", methods=["GET"])
def dash():
    return render_template("dashboard.html")


# ================= ADMIN ROUTES =================


# Admin dashboard to manage users.
@app.route("/admin", methods=["GET", "POST"])
def admin():
    data = db.getusers()
    if request.method == "POST":
        if "delete_username" in request.form:
            db.deleteUser(request.form["delete_username"])
            data = db.getusers()
            return render_template(
                "admin.html",
                active_class="admin",
                patient_id=db.getPatientsNumber(),
                data=data,
                data_len=len(data),
            )
        if "edit_username" in request.form:
            db.editUser((request.form["edit_user_type"], request.form["edit_username"]))
            data = db.getusers()
            return render_template(
                "admin.html",
                active_class="admin",
                patient_id=db.getPatientsNumber(),
                data=data,
                data_len=len(data),
            )
        username = request.form["username"]
        user_email = request.form["user_email"]
        password = request.form["password"]
        user_type = request.form["user_type"]
        success = db.addusers(
            (
                user_email,
                user_type,
                username,
                hashlib.sha256(escape(password).encode()).hexdigest(),
            )
        )
        if success:
            data = db.getusers()
            return render_template(
                "admin.html",
                active_class="admin",
                patient_id=db.getPatientsNumber(),
                success=True,
                data=data,
                data_len=len(data),
            )
        else:
            return render_template(
                "admin.html",
                active_class="admin",
                patient_id=db.getPatientsNumber(),
                failed=True,
                data=data,
                data_len=len(data),
            )

    return render_template(
        "admin.html",
        active_class="admin",
        patient_id=db.getPatientsNumber(),
        data=data,
        data_len=len(data),
    )


# ================= DOCTOR ROUTES =================


# Add new doctors and display existing doctor details.
@app.route("/doctors", methods=["GET", "POST"])
def doctors():
    ddata = db.getdoctor_details()
    try:
        if request.method == "POST":
            doctor_id = request.form["doc_id"]
            doctor_name = request.form["doc_name"]
            doctor_designation = request.form["doc_des"]
            doctor_department = request.form["doc_dep"]
            doctor_email = request.form["doc_email"]
            doctor_phnno = request.form["doc_phnno"]
            doctor_hospital = request.form["doc_hospital"]
            doc_id = db.addnewdoc_details(
                (
                    doctor_name,
                    doctor_designation,
                    doctor_department,
                    doctor_email,
                    doctor_phnno,
                    doctor_hospital,
                )
            )
            if doc_id:
                ddata = db.getdoctor_details()

                return render_template(
                    "doctors.html",
                    active_class="doctors",
                    success=True,
                    data=ddata,
                    data_len=len(ddata),
                )
            else:
                return render_template(
                    "doctors.html",
                    active_class="doctors",
                    failed=True,
                    data=ddata,
                    data_len=len(ddata),
                )
    except Exception as e:
        print(e)
    return render_template(
        "doctors.html", active_class="doctors", data=ddata, data_len=len(ddata)
    )


# ================= PATIENT ROUTES =================


# Add new patients and list all patients.
@app.route("/patients", methods=["GET", "POST"])
def patient():
    p_id = None
    pdata1 = db.getpatientdetails()
    doc_ids = db.getlast_docid()
    if request.method == "POST":
        patient_name = request.form["patient_name"]
        patient_email = request.form["patient_email"]
        patient_mobile = request.form["patient_mobile"]
        patient_gender = request.form["patient_gender"]
        patient_DOB = request.form["patient_DOB"]
        patient_age = request.form["patient_age"]
        patient_address = request.form["patient_address"]
        patient_blood_group = request.form["patient_bloodgroup"]
        patient_docid = request.form["reff_docid"]
        p_id = db.addnewpatients(
            (
                patient_name,
                patient_email,
                patient_mobile,
                patient_gender,
                patient_DOB,
                patient_address,
                patient_blood_group,
            ),
            patient_docid,
        )
        if p_id > 0:
            pdata1 = db.getpatientdetails()
            return render_template(
                "patients.html",
                active_class="patients",
                patient_id=p_id,
                success=True,
                data=pdata1,
                data_len=len(pdata1),
            )
        else:
            return render_template(
                "patients.html",
                active_class="patients",
                patient_id=p_id,
                failed=True,
                data=pdata1,
                data_len=len(pdata1),
            )
    return render_template(
        "patients.html",
        active_class="patients",
        data=pdata1,
        data_len=len(pdata1),
        patient_id=p_id,
    )


# ================= REPORT & EVALUATION ROUTES =================


# Upload images, run preprocessing, and save to database.
@app.route("/bst", methods=["GET", "POST"])
def bst():
    folder_name = ""
    param_path = ""
    if request.method == "POST":
        # Step 1: fetch patient details if patient_id submitted
        if "patient_id" in request.form:
            patient_id = request.form["patient_id"]
            pdetails = db.getpatient_details(patient_id)
            if len(pdetails) > 0:
                pdata = [pdetails[0][0], pdetails[0][1], pdetails[0][2], pdetails[0][3]]
                return render_template(
                    "bst.html", active_class="bst", data=pdata, success=True
                )
            else:
                return render_template("bst.html", active_class="bst", failed=True)
        else:
            # Step 2: if images are uploaded
            uploaded_files = request.files.getlist("rbc_images[]")
            pid = request.form["patient_id_hidden"]
            UPLOAD_FOLDER = "E:/Project25/J_Blosmia1/static/images/styles/pre"
            uploaded_folder = "E:/Project25/J_Blosmia1/static/images/styles/a/"

            # Get new image count and create unique folder name
            count = db.imagecount()
            img_count = str(count + 1)
            folder_name = pid + "_" + img_count  # Folder name
            param_path = os.path.join(uploaded_folder, folder_name)  # Full path on disk
            os.makedirs(param_path, exist_ok=True)

            # Relative path for DB (as required)
            db_folder_path = "images/styles/a/" + folder_name

            # Insert report
            data = [pid, timeStamp]
            rep_id = db.report(data)

            # Track processing status
            all_success = True

            # Save and move uploaded images
            img_ids = []
            for image_file in uploaded_files:
                # Save temporarily in pre folder
                temp_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
                image_file.save(temp_path)

                # Move image to target folder
                destination_path = os.path.join(param_path, image_file.filename)
                shutil.move(temp_path, destination_path)

                # Read image binary for DB insert
                with open(destination_path, "rb") as f:
                    img_blob = f.read()

                # Insert image into DB with relative folder path
                img_id = db.addimage(
                    rep_id, image_file.filename, img_blob, db_folder_path
                )
                img_ids.append(img_id)

                # Call preprocess for this image
                success = pre.preprocess(destination_path, param_path, img_id, rep_id)
                if not success:
                    all_success = False

            # Call preprocessing on the folder with all uploaded images
            # predict = pre.preprocess(param_path, param_path, img_ids[-1], rep_id)

            # After all images processed → update report result table
            if all_success:
                types_a = db.get_total_count(rep_id)
                cell_count_data = [
                    rep_id,
                    types_a[0][0],
                    types_a[1][0],
                    types_a[2][0],
                    types_a[3][0],
                    types_a[4][0],
                    types_a[5][0],
                    types_a[6][0],
                    types_a[7][0],
                ]
                db.add_count_in_reportResult(cell_count_data)
                return render_template("bst.html", active_class="bst", predict=True)
            else:
                return render_template("bst.html", active_class="bst", predict=False)
    return render_template("bst.html", active_class="bst")


# Show evaluation data for analysis.
@app.route("/evaluate", methods=["GET", "POST"])
def evaluation():
    if request.method == "POST":
        if "patient_id" in request.form:
            patient_id = request.form["patient_id"]
            details = db.getpatient_details(patient_id)
            if len(details) > 0:
                data = [details[0][0], details[0][1], details[0][2], details[0][3]]
                rbcData = db.getrbcData(patient_id)
                if len(rbcData) >= 0:
                    return render_template(
                        "evaluation.html",
                        active_class="evaluate",
                        data=data,
                        success=True,
                        rbcData=rbcData,
                        rbcDataLen=len(rbcData),
                        report=rbcData,
                    )
                else:
                    return render_template(
                        "evaluation.html",
                        active_class="evaluate",
                        data=data,
                        success=True,
                    )
            else:
                return render_template(
                    "dashboard.html", active_class="evaluate", failed=True
                )
    return render_template("evaluation.html", active_class="evaluate")


# Serve images from TrainOutput folder.
@app.route("/display/<filename>")
def display_image(filename):
    return redirect(
        url_for("static", filename="images/styles/a" + "/" + filename), code=301
    )


# Detailed report view with blob images and editable classifications.
@app.route("/detailedrep", methods=["GET", "POST"])
def detailedreport():  # call the new function get_total_count
    patient_id = ""
    image_id = ""
    images = ""
    blob_ids = ""
    n = ""
    repid = ""
    splited_blob_id = []
    types_a = []
    image_list = []
    if request.method == "POST":
        if "patient_id" in request.form:
            repid = int(request.form["report_id"])
            patient_id = int(request.form["patient_id"])
            test_date = request.form["test_date"]
        if "edit_blob_id" in request.form:
            repid = int(request.form["report_id"])
            edit_id = request.form["edit_blob_id"]
            rbctype = request.form["rbctype"]
            # patient_id = request.form['patient_id']
            res = db.editreport(edit_id, rbctype)
            types_a = db.get_total_count(repid)
            cell_count_data = [
                types_a[0][0],
                types_a[1][0],
                types_a[2][0],
                types_a[3][0],
                types_a[4][0],
                types_a[5][0],
                types_a[6][0],
                types_a[7][0],
                repid,
            ]
            db.update_report_result(cell_count_data)
        if patient_id:
            details = db.getpatient_details(patient_id)
            if len(details) > 0:
                data = [
                    details[0][0],
                    details[0][1],
                    details[0][2],
                    details[0][3],
                    test_date,
                    image_id,
                    patient_id,
                ]
                blob_data = db.get_blob_data_to_display(repid)
                typesa = db.get_report_result_data(repid)
                wbc_type = [
                    typesa[0][0],
                    typesa[0][2],
                    typesa[0][3],
                    typesa[0][4],
                    typesa[0][5],
                ]
                rbc_type = [typesa[0][1], typesa[0][6], typesa[0][7]]
                blob_ids = [row[0] for row in blob_data]
                blob_image = [row[1] for row in blob_data]
                blob_name = [row[2] for row in blob_data]
                blob_cell_id = [row[3] for row in blob_data]
                # print(len(blob_name))
                for img in blob_image:
                    # Convert BLOB data to a NumPy array
                    nparr = np.frombuffer(img, np.uint8)
                    # Decode the NumPy array to an image
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    success, buffer = cv2.imencode(".png", image)
                    # Convert buffer to base64 string
                    if success:
                        converted_blob = base64.b64encode(buffer).decode("utf-8")
                        image_list.append(converted_blob)
                images_with_index = list(enumerate(image_list))
            return render_template(
                "detailedreport.html",
                active_class="evaluate",
                data=data,
                blob_name=blob_name,
                success=True,
                datalength=len(image_list),
                image_list=images_with_index,
                blob_ids=blob_ids,
                blob_cell_id=blob_cell_id,
                repid=repid,
                wbc_type=wbc_type,
                rbc_type=rbc_type,
            )
        else:
            return render_template(
                "detailedreport.html", active_class="evaluate", rbcfailed=True
            )
    return render_template("detailedreport.html", active_class="evaluate")


# Final summarized report for patient.
data1 = []


@app.route("/final", methods=["GET", "POST"])
def final():
    if request.method == "POST":
        if "patient_id" in request.form:
            patient_id = request.form["patient_id"]
            repid = request.form["repid"]
            details = db.getpatientdetail(patient_id)
            if len(details) > 0:
                newdata = db.get_report_result_data(repid)
                # print(newdata)
                if len(newdata) > 0:
                    data = [
                        details[0][0],
                        details[0][1],
                        details[0][2],
                        details[0][3],
                        details[0][4],
                        details[0][5],
                        details[0][6],
                        details[0][7],
                    ]
                    data1 = [
                        newdata[0][0],
                        newdata[0][1],
                        newdata[0][2],
                        newdata[0][3],
                        newdata[0][4],
                        newdata[0][5],
                        newdata[0][6],
                        newdata[0][7],
                    ]
                    return render_template(
                        "finalreport.html",
                        active_class="final",
                        data=data,
                        data1=data1,
                        success=True,
                    )
                else:
                    # No report available for the given date
                    return render_template(
                        "finalreport.html", active_class="final", nodata=True
                    )
            else:
                # Patient details not found
                return render_template(
                    "finalreport.html", active_class="final", failed=True
                )
        else:
            # Patient ID or Date not entered correctly
            return render_template(
                "finalreport.html", active_class="final", failed=True
            )
    return render_template("finalreport.html", active_class="final")


# Generate invoice PDF for a patient report.
@app.route("/invoice/<int:invoice_id>/pdf")
def generate_invoice_pdf(invoice_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "Select basophil,double_rbc,eosinophil,Lymphocyte,Monocyte,Neutrophil,single_rbc,triple_rbc from ReportResult RR inner JOIN report rp on RR.report_id = rp.report_id where rp.patient_id=?",
        [invoice_id],
    )
    invoice = cursor.fetchall()
    conn.close()
    details = db.getpatientdetail(invoice_id)
    data = [
        details[0][0],
        details[0][1],
        details[0][2],
        details[0][3],
        details[0][4],
        details[0][5],
        details[0][6],
        details[0][7],
    ]
    newdata = db.pdf(invoice_id)
    data11 = [
        newdata[0][0],
        newdata[0][1],
        newdata[0][2],
        newdata[0][3],
        newdata[0][4],
        newdata[0][5],
        newdata[0][6],
        newdata[0][7],
        newdata[0][8],
        newdata[0][9],
        newdata[0][10],
        newdata[0][11],
        newdata[0][12],
        newdata[0][13],
    ]
    path_to_wkhtmltopdf = (
        r"E:/Project25/J_Blosmia1/static/images/styles/wkhtmltopdf/bin/wkhtmltopdf.exe"
    )
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    rendered_html = render_template("invoice.html", data1=data11, data=data)
    pdf = pdfkit.from_string(rendered_html, False, configuration=config)
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "inline; filename=report.pdf"
    return response


# ================= TRAINING & RETRAINING ROUTES =================


# Retraining page 1.
@app.route("/retrain1", methods=["GET"])
def retrain1():
    return render_template("retrain1.html")


# Retraining page 2.
@app.route("/retrain2", methods=["GET"])
def retrain2():
    return render_template("retrain2.html")


# Retraining page 3 (step 2).
@app.route("/retrain32", methods=["GET"])
def retrain32():
    return render_template("retrain32.html")


# Upload files for retraining.
@app.route("/retrain3", methods=["POST"])
def upload_file():
    if "bwfile" not in request.files:
        return redirect(request.url)
    uploaded_files = request.files.getlist("bwfile")
    basepath = os.path.dirname(__file__)
    for image_file in uploaded_files:
        if request.method == "POST":
            image_file.save(os.path.join(TRAIN_OUTPUT_FOLDERNAME, image_file.filename))
            if image_file:
                f = image_file.filename
            img = image_file.read()
            predict = tp.preprocess(TRAIN_OUTPUT_FOLDERNAME)
            FILE = image_file.filename
            if f == FILE:
                os.remove(os.path.join(TRAIN_OUTPUT_FOLDERNAME, FILE))
                print(f"Image '{FILE}' removed successfully")
    if predict:
        return redirect(url_for("retrain2"))


# Move blobs to training set and update database after retraining.
@app.route("/modelretrain", methods=["GET", "POST"])
def modelretrain():
    blob_data = db.get_blob_to_trained()
    for blobdata in blob_data:
        cellsid, blobimg = blobdata
        res = db.inserttested(blobimg, cellsid)
    if res:
        db.update_blob_after_train()
    # add code to put these images in the folder
    return render_template("retrain33.html")


@app.route("/modelretrain2", methods=["GET", "POST"])

# Save retraining images and retrain model with WBC images.
def modelretrain2():
    blob_retreive = db.get_blob_retrain()
    OUTPUT_DIR = "E:/Project25/J_Blosmia1/static/training/retraining_model"
    for index, row in enumerate(blob_retreive):
        cell_image, cellsid = (
            row  # Retrieve cellsid, binary image data, and image filename
        )
        # Create a folder for each cellsid
        folder_path = os.path.join(
            OUTPUT_DIR, str(cellsid)
        )  # Folder for the specific cellsid
        os.makedirs(folder_path, exist_ok=True)
        # Ensure a unique filename using indexing
        existing_files = os.listdir(folder_path)  # List of existing files in the folder
        index = len(existing_files) + 1  # Start from the next available index
        image_name = f"image_{index}.jpg"  # Generate a filename using the updated index
        image_path = os.path.join(folder_path, image_name)
        # Debugging logs
        print(f"Saving image {image_name} for cellsid {cellsid} to {image_path}")
        # Save the binary image data to a file
        with open(image_path, "wb") as img_file:
            img_file.write(cell_image)  # Write the binary image data
    # Call the model training function after saving images
    m.modeltrain()
    # Return to the template
    return render_template("retrain33.html")


# ================= MAIN =================

if __name__ == "__main__":
    [app.run(debug=True)]
