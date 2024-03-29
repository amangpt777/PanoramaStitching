<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Panorama</title>
  <script type="text/javascript" src="lib/jquery-2.js"></script>
  <script type="text/javascript" src="lib/jquery.panorama_viewer.js"></script>
  <script type="text/javascript">
   function toggle_visibility(id) {
       var e = document.getElementById(id);
       if(e.style.display == 'block')
          e.style.display = 'none';
       else
          e.style.display = 'block';
   }
</script>
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

  <div class="wrapper">
    <div class="main">
      <div class="header">
        <h1 class="title"></h1>
        <h2 class="author"></h2>
      </div>
      <div class="content">
        <div class="panorama">
         <img class="image" src="./Output1/cylindrical_panorama.png">
        </div>
      </div>
    </div>
  </div>
  
</body>
</html>


