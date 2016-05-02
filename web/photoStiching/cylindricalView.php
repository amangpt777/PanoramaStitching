<!doctype html>
<html>
<body>
<?php
	
	$output = exec('sudo sh runTestWebApp.sh', $out);
	
?>


<form enctype="multipart/form-data" action="view.php" method="post" name="upload-form" id="upload-form">

 <input type="submit" value="View Stitch!" id = "view" style="position: relative; left: 250px; top: 100px;"/>

</form>
	
</body>
</html>