<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Panorama</title>
  <script type="text/javascript" src="lib/jquery-2.js"></script>
  <script type="text/javascript" src="lib/jquery.panorama_viewer.js"></script>
  <link href='lib/panorama_viewer.css' rel='stylesheet' type='text/css'>
  <style>
    body {
      background: #F1f1f2;
      margin: 0;
    }
    .header {
      z-index: 2;
      position: fixed;
      top: 0;
      width: 100%;
      height: 105px;
      font-family: sans-serif;
      text-align: center;
    }
    .title {
      color: #333;
      font-size: 40px;
      line-height: 40px;
      font-weight: 100;
      padding-top: 20px;
      padding-bottom: 0;
      margin: 0;
    }
    .author {
      color: #666;
      font-size: 20px;
      line-height: 20px;
      font-weight: 100;
      padding-top: 5px;
      padding-bottom: 20px;
      margin: 0;
    }
    .content {
      z-index: 1;
      width: 100%;
      height: 100vh;
      padding: 0;
      margin: 0;
    }
    .panorama {
      position: absolute;
      bottom: 0;
      height: -moz-calc(100% - 105px);
      height: -webkit-calc(100% - 105px);
      height: calc(100% - 105px);
      width: 100%;
    }
  </style>
  <script>
  // Use $(window).load() on live site instead of document ready. This is for the purpose of running locally only
    $(document).ready(function(){
      $(".panorama").panorama_viewer({
        repeat: true,
        direction: "horizontal", 
        animationTime: 400,  
        easing: "ease-out",        
        overlay: true
      });
    });

    var rtime = new Date(1, 1, 2000, 12, 00, 00);
    var timeout = false;
    var delta = 200;
    $(window).resize(function() {
        rtime = new Date();
        if (timeout === false) {
            timeout = true;
            setTimeout(resizeend, delta);
        }
    });
    function resizeend() {
        if (new Date() - rtime < delta) {
            setTimeout(resizeend, delta);
        } else {
            timeout = false;
            location.reload();
        }               
    }
  </script>
</head>
<body>


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
					$output = exec('sudo sh runTestWebApp.sh', $out);

				} else
				{
					echo  "Errors occoured during file upload!";
				}
			}	

		}
	}
	
	
	
	
?>

  <div class="wrapper">
    <div class="main">
      <div class="header">
        <h1 class="title"></h1>
        <h2 class="author"></h2>
      </div>
      <div class="content">
        <div class="panorama">
         <img class="image" src="./Output/mountain_panorama.png">
        </div>
      </div>
    </div>
  </div>
</body>
</html>
