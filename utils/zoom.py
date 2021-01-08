import os

os.system("set AFNI_NIFTI_TYPE_WARN = NO")
os.system("mkdir 2x")
for idx in range(10):
	idx_str = "{0:0>3}".format(idx+1)
	print(idx_str)
	cmd_1 = "3dresample -dxyz 0.586 0.586 1.39 -prefix z_2x_"+idx_str+" -inset z"+idx_str+".nii -rmode Cu"
	# cmd_2 = "3dZeropad -I 17 -S 17 p"+idx_str+"+orig"
	cmd_3 = "3dAFNItoNIFTI -prefix z_2x_"+idx_str+" z_2x_"+idx_str+"+orig"
	# cmd_4 = "rm -f zeropad+orig.BRIK"
	# cmd_5 = "rm -f zeropad+orig.HEAD"
	cmd_6 = "rm -f z_2x_"+idx_str+"+orig.BRIK"
	cmd_7 = "rm -f z_2x_"+idx_str+"+orig.HEAD"
	cmd_8 = "gzip z_2x_"+idx_str+".nii"
	cmd_9 = "mv z_2x_"+idx_str+".nii.gz ./2x/"
	# cmd_6 = "mv y"+idx_str+".nii ../inv_RSZP"
	for cmd in [cmd_1, cmd_3, cmd_6, cmd_7, cmd_8, cmd_9]:
		print(cmd)
		os.system(cmd)
# 3dresample -dxyz 1.172 1.172 2.78 -prefix test -inset BraTS20_Training_001_t1_inv.nii
# 3dZeropad -I 16 -S 17 -A 25 -P 26 -L 25 -R 26 Z001+orig -prefix 123
# 3dAFNItoNIFTI -prefix test test+orig