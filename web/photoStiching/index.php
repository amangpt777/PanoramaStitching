<?php
	$up_id = uniqid(); 
?>


<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=utf-8" />
<title>Panorama Stitching</title>
<meta name="description" content="Free PHP multi file uploader script with progress bar which is easy to configure and use." />
<meta http-equiv="Language" content="en" />
<meta name="language" content="en" />
<!--Progress Bar and iframe Styling-->
<link href="style-progress.css" rel="stylesheet" type="text/css" />

<!--Get jQuery-->
<!-- script src="http://ajax.googleapis.com/ajax/libs/jquery/1.4.0/jquery.js" type="text/javascript"></script -->
<script src="lib/jquery.js" type="text/javascript"></script>
<script language="JavaScript" type="text/javascript">
// allow all extensions
var exts = "";
// only allow specific extensions
// var exts = "jpg|jpeg|gif|png|bmp|tiff|pdf";
function checkExtension(value)
{
    if(value=="")return true;
    var re = new RegExp("^.+\.("+exts+")$","i");
    if(!re.test(value))
    {
        alert("Your file extension is not allowed: \n" + value + "\n\nOnly the following extensions are allowed: "+exts.replace(/\|/g,',')+" \n\n");
        return false;
    }
    return true;
}
$(document).ready(function() { 
//
//show the progress bar only if a file field was clicked
	var show_bar = 0;
    $('input[type="file"]').click(function(){
		show_bar = 1;
    });
//show iframe on form submit
    $("#upload-form").submit(function(){
		if (show_bar === 1) { 
			$('#progress-frame').show();
			function set () {
				$('#progress-frame').attr('src','progress-frame.php?up_id=<?php echo $up_id; ?>');
			}
			setTimeout(set);
		}
    });
//
});
var next_id=0;
var max_number =20;
	function _add_more() {
		
		if (next_id>=max_number)
		{
			alert("You reached maximum number of 20 files!");
			return;
		}
		next_id=next_id+1;
		var next_div=next_id+1;
		var txt = "<br><input type=\"file\" name=\"item_file[]\" onChange=\"checkExtension(this.value)\">";
		txt+='<div id="dvFile'+next_div+'"></div>';
		document.getElementById("dvFile" + next_id ).innerHTML = txt;
	}
	function validate(f){
		var chkFlg = false;
		for(var i=0; i < f.length; i++) {
			if(f.elements[i].type=="file" && f.elements[i].value != "") {
				chkFlg = true;
			}
		}
		if(!chkFlg) {
			alert('Please browse/choose at least one file');
			return false;
		}
		f.pgaction.value='upload';
		return true;
	}
</script>


</head>

<body>
<p><b>Upload the files you want to stitch</b></p>

<form enctype="multipart/form-data" action="upload.php" method="post" name="upload-form" id="upload-form">

        <!--hidden field-->
         <input type="hidden" value="demo" name="<?php echo ini_get("session.upload_progress.name"); ?>"/>
        <!---->


	<div id="dvFile0"><input type="file" name="item_file[]" onChange="checkExtension(this.value)"></div><div id="dvFile1"></div>
        <a href="javascript:_add_more(0);"><B>(+) Add file</B></a>
	<input type="submit" value="Upload!">
</form>
	
	<!--Include the progress bar frame-->
   	 <iframe style="position: relative; top: 5px;" id="progress-frame" name="progress-frame" border="0" src="" scrollbar="no" frameborder="0" scrolling="no"> </iframe>
	<!---->
</body>
</html> 