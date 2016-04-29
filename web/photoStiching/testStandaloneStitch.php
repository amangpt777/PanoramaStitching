<!DOCTYPE html>
<html>
<body>

<h1>My first PHP page</h1>

<?php
echo "Hello World1!";
//$output = exec('python /Applications/XAMPP/htdocs/testWeb.py', $out);
//$output = exec('/Applications/MATLAB_R2014b.app/bin/matlab', $out);
//$output = exec('/Applications/MATLAB_R2014b.app/bin/matlab -nosplash -nodesktop -logfile out1.txt -r testWeb', $out);
$output = exec('sudo sh runTestWebApp.sh', $out);
sleep(5);
echo $output;
foreach($out as $key => $value)
{
    echo $key." ".$value."<br>";
}


echo "Done1";
?>

</body>
</html>
