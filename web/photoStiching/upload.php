<?php
	
	if (!isset($_FILES["item_file"]))
		die ("Error: no files uploaded!");

	$file_count = count($_FILES["item_file"]['name']);
	
	echo $file_count . " file(s) sent... <BR><BR>";

	if(count($_FILES["item_file"]['name'])>0) { //check if any file uploaded

		for($j=0; $j < count($_FILES["item_file"]['name']); $j++) { //loop the uploaded file array
			
			$filen = $_FILES["item_file"]['name'][$j];	

			// ingore empty input fields
			if ($filen!="")
			{
		
				// destination path - you can choose any file name here (e.g. random)
				$path = "upload/" . $filen; 

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

