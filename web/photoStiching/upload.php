<?php
	
	if (!isset($_FILES["item_file"]))
		die ("Error: no files uploaded!");

	$file_count = count($_FILES["item_file"]['name']);
	
	echo $file_count . " file(s) sent... <BR><BR>";

	if(count($_FILES["item_file"]['name'])>0) { //check if any file uploaded

		for($j=0; $j < count($_FILES["item_file"]['name']); $j++) { //loop the uploaded file array
			
			$temp = explode(".", $_FILES["item_file"]['name'][$j]);
			$newfilename = 'image_'. ($j+1) . '.' . end($temp);   // rename the file with order number
			$filen = $_FILES["item_file"]['name'][$j];	

			// ingore empty input fields
			if ($filen!="")
			{
		
				// destination path - you can choose any file name here (e.g. random)
				//$path = "upload/" . $filen; 
				$path = "upload/" . $newfilename;
                
                if ($j == 0) {
                    array_map('unlink', glob('upload/*'));   // delete all the files in upload directory
                }
                
				if(move_uploaded_file($_FILES["item_file"]['tmp_name']["$j"],$path)) { 
				
					echo "File# ".($j+1)." ($filen) uploaded successfully!<br>"; 

				} else
				{
					echo  "Errors occoured during file upload!";
				}
			}	

		}
	}
	
?>

