$(document).ready(function () {
    // Hide sections initially
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview Function
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

    // Handle Image Upload
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict Button Click Event
    $('#btn-predict').click(function (event) {
        event.preventDefault(); // Prevent page reload
        
        var form_data = new FormData($('#upload-file')[0]);

        // Show loader animation
        $(this).hide();
        $('.loader').show();
        $('#result').html("<p class='text-warning'>Processing...</p>");
        $('#result').show();

        // AJAX call to Flask API
        $.ajax({
            type: "POST",
            url: "/predict",
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (response) {
                $('.loader').hide();
                $('#btn-predict').show();

                if (response.error) {
                    $("#result").html(`<p class="text-danger">${response.error}</p>`);
                } else {
                    let predictionText = `
                        <h4>Prediction: <span class="text-primary">${response.class}</span></h4>
                        <h5>Confidence: <span class="text-success">${response.confidence}%</span></h5>
                    `;
                    $("#result").html(predictionText);
                }
            },
            error: function (xhr) {
                $('.loader').hide();
                $('#btn-predict').show();
                $("#result").html(`<p class="text-danger">Error: ${xhr.responseText || "Server error"}</p>`);
            }
        });
    });
});
