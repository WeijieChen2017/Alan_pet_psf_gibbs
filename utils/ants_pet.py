import ants

idx_list = ["001", "002", "005", "010", "019", "041", "046", "053", "057", "058", "068", "070"]
for idx in idx_list:
    print("%"*50)
    name_mri = "sCT_"+idx+"_MRI.nii.gz"
    name_pet = "sCT_"+idx+"_PET.nii.gz"
    name_rs = "sCT_"+idx+"_ants.nii.gz"
    file_mri = ants.image_read(name_mri)
    file_pet = ants.image_read(name_pet)
    rs_pet = ants.resample_image_to_target( image=file_pet, target=file_mri, type='bSpline' )
    ants.image_write( rs_pet, name_rs)
    print(name_rs)
