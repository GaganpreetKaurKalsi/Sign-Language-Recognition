let predicted = "Choose an image to predict"
let video = document.getElementById('video')
var form = document.getElementById("img-form");
var realData;

$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result-section').hide();
    $('.take-section').hide();
    $('#btn-predict-img').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    function stopVideo() {
        var stream = video.srcObject;
        var tracks = stream.getTracks();
        for (var i = 0; i < tracks.length; i++) {
            var track = tracks[i];
            track.stop();
        }
        video.srcObject = null;
    }

    function startVideo() {
         vendorUrl = window.URL || window.webkitURL;
         if (navigator.mediaDevices.getUserMedia) {
             navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
                video.srcObject = stream;
                document.querySelector(".btn-take-pic").disabled = false;
                document.querySelector(".stop-btn").disabled = false;
                document.querySelector(".start-btn").disabled = true;
             }).catch(function (error) {
                console.log("Something went wrong!");
             });
         }
    }

    function takeSnapshot() {
      var img = document.querySelector('.img-display').querySelector('img')
      console.log(img)
      var context;
      var width = video.offsetWidth
        , height = video.offsetHeight;

      var canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;

      context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, width, height);

      img.src = canvas.toDataURL('image/png');
      document.querySelector('.img-display').appendChild(img);
      $('#btn-predict-img').show();

      var ImageURL = img.src; // 'photo' is your base64 image
      // Split the base64 string in data and contentType
      var block = ImageURL.split(";");
      // Get the content type of the image
      var contentType = block[0].split(":")[1];// In this case "image/gif"
      // get the real base64 content of the file
      realData = block[1].split(",")[1];
    }


    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result-section').hide();
        $('#btn-predict-img').hide();
        readURL(this);
    });

    $(".upload-btn").click(function () {
        $('.take-section').hide();

    });

    $(".click-photo").click(function () {
        $('.image-section').hide();
        $('#btn-predict').hide();
        $('#result').text('');
        $('#result-section').hide();
        $('.take-section').show();
    });

    $("#btn-clear").click(function () {
        $('.image-section').hide();
        $('#btn-predict').hide();
        $('#result').text('');
        $('#result-section').hide();
    });

    $("#click-photo").click(function () {
        $('.video-section').show();
    });

    $(".btn-take-pic").click(function () {
        takeSnapshot()
    });

    $(".start-btn").click(function () {
        startVideo()
    });

    $(".stop-btn").click(function () {
        stopVideo()
        document.querySelector(".btn-take-pic").disabled = true;
        document.querySelector(".stop-btn").disabled = true;
        document.querySelector(".start-btn").disabled = false;
    });

    $("#speaker").click(function () {
        var msg = new SpeechSynthesisUtterance();
        msg.text = predicted;
        window.speechSynthesis.speak(msg);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        console.log(form_data)

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result-section').fadeIn(600);
                $('#result').text(data);
                predicted = data
                console.log('Success!');
            },
        });
    });

    // Predict
    $('#btn-predict-img').click(function () {
        // Show loading animation
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict-img',
            data: realData,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result-section').fadeIn(600);
                $('#result').text(data);
                predicted = data
                console.log('Success!');
            },
        });

    });

});