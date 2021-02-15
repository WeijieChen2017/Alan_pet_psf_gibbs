import os

os.system("set AFNI_NIFTI_TYPE_WARN = NO")
os.system("mkdir PET_2x")
idx_list = ["001", "002", "005", "010", "019", "041", "046", "053", "057", "058", "068", "070"]
for idx_str in idx_list:
	# idx_str = "{0:0>3}".format(idx+1)
	print(idx_str)
	cmd_1 = "3dresample -dxyz 1.172 1.172 1.07 -prefix pet"+idx_str+" -inset sCT_"+idx_str+"_PET.nii.gz -rmode Cu"
	# cmd_2 = "3dZeropad -I 17 -S 17 p"+idx_str+"+orig"
	cmd_3 = "3dAFNItoNIFTI -prefix pet"+idx_str+" pet"+idx_str+"+orig"
	# cmd_4 = "rm -f zeropad+orig.BRIK"
	# cmd_5 = "rm -f zeropad+orig.HEAD"
	cmd_6 = "rm -f pet"+idx_str+"+orig.BRIK"
	cmd_7 = "rm -f pet"+idx_str+"+orig.HEAD"
	cmd_8 = "mv p"+idx_str+".nii ./PET_2x/"
	# cmd_6 = "mv y"+idx_str+".nii ../inv_RSZP"
	for cmd in [cmd_1, cmd_3, cmd_6, cmd_7, cmd_8]:
		print(cmd)
		os.system(cmd)
# 3dresample -dxyz 1.172 1.172 2.78 -prefix test -inset BraTS20_Training_001_t1_inv.nii
# 3dZeropad -I 16 -S 17 -A 25 -P 26 -L 25 -R 26 Z001+orig -prefix 123
# 3dAFNItoNIFTI -prefix test test+orig