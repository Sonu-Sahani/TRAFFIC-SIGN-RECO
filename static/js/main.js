$(document).ready(function () {
    let videoStream = null;
    let cameraInterval = null;

    // ========== IMAGE UPLOAD ==========
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide().fadeIn(500);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#image-result').html('');
        readURL(this);
    });

    $('#btn-predict').click(function (event) {
        event.preventDefault();
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();
        $('#image-result').html('<p class="text-warning">Processing...</p>');

        $.ajax({
            type: "POST",
            url: "/predict",
            data: form_data,
            contentType: false,
            processData: false,
            success: function (response) {
                $('.loader').hide();
                $('#btn-predict').show();
                if (response.error) {
                    $('#image-result').html(`<p class="text-danger">${response.error}</p>`);
                } else {
                    let badge = response.confident ? '<span class="badge badge-success">High Confidence</span>' : '<span class="badge badge-warning">Low Confidence</span>';
                    $('#image-result').html(`
                        <h4>${response.class} ${badge}</h4>
                        <h5>Confidence: <span class="text-primary">${response.confidence}%</span></h5>
                    `);
                }
            },
            error: function () {
                $('.loader').hide();
                $('#btn-predict').show();
                $('#image-result').html('<p class="text-danger">Server error</p>');
            }
        });
    });

    // ========== LIVE CAMERA ==========
    $('#start-camera').click(async function () {
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
            document.getElementById('video').srcObject = videoStream;
            $('#start-camera').hide();
            $('#stop-camera').show();
            
            // Predict every 800ms
            cameraInterval = setInterval(captureAndPredict, 800);
        } catch (err) {
            alert('Camera access denied or not available');
            console.error(err);
        }
    });

    $('#stop-camera').click(function () {
        if (cameraInterval) clearInterval(cameraInterval);
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
        }
        $('#stop-camera').hide();
        $('#start-camera').show();
        $('#camera-result').html('');
    });

    function captureAndPredict() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const dataURL = canvas.toDataURL('image/jpeg', 0.7);

        $.ajax({
            type: 'POST',
            url: '/predict_frame',
            contentType: 'application/json',
            data: JSON.stringify({ image: dataURL }),
            success: function (res) {
                if (res.error) return;
                let badge = res.confident ? '<span class="badge badge-success">High</span>' : '<span class="badge badge-secondary">Low</span>';
                $('#camera-result').html(`
                    <h5>${res.class} ${badge}</h5>
                    <p>Confidence: ${res.confidence}%</p>
                `);
            }
        });
    }

    // ========== VIDEO UPLOAD ==========
    $('#videoUpload').change(function () {
        $('#btn-video-predict').show();
        $('#video-result').html('');
    });

    $('#btn-video-predict').click(function () {
        var form_data = new FormData($('#video-form')[0]);
        $(this).hide();
        $('#video-loader').show();
        $('#video-result').html('<p class="text-warning">Analyzing video...</p>');

        $.ajax({
            type: 'POST',
            url: '/predict_video',
            data: form_data,
            contentType: false,
            processData: false,
            success: function (res) {
                $('#video-loader').hide();
                $('#btn-video-predict').show();
                if (res.error) {
                    $('#video-result').html(`<p class="text-danger">${res.error}</p>`);
                } else if (res.most_common_class) {
                    $('#video-result').html(`
                        <h4>Most Detected: <strong>${res.most_common_class}</strong></h4>
                        <p>Detected in ${res.count} frames | Avg Confidence: ${res.avg_confidence}%</p>
                        <p>Total confident detections: ${res.total_detections} / ${res.frames_processed} frames</p>
                    `);
                } else {
                    $('#video-result').html(`<p>${res.message || 'No confident signs detected'}</p>`);
                }
            },
            error: function () {
                $('#video-loader').hide();
                $('#btn-video-predict').show();
                $('#video-result').html('<p class="text-danger">Error processing video</p>');
            }
        });
    });
});