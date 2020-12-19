$(document).ready(function() {

  let currently_generating = false;
  let setup = false;
  $("#start-btn").on('click', function() {

    if (!setup) {
      setup = true;
      $("#hamburger").fadeIn(500);
    } else {
      $("#hamburger .ham").removeClass("active");
    }
    $("#setup").fadeOut(500);
    let token_vals = [];
    ($("input:checked")).each(function() {
      token_vals.push("["+$(this).val()+"]");
    });
    let blacklist_vals = [];
    $(".blacklist").each(function() {
      blacklist_vals.push($(this).data("word"))
    });
    let current_story = $("#current-story").text();
    let temperature = $("#temperature").val();
    let generationLength = $("#generation-length").val();
    let story_json = JSON.stringify({
      blacklist: blacklist_vals,
      tokens: token_vals,
      context: current_story,
      temperature: temperature,
      generation_length: generationLength
    });
    $.ajax({
      data: story_json,
      type: 'POST',
      url: "/settings",
      contentType: 'application/json;charset=UTF-8',
      success: function(dataofconfirm) {
        // do something with the result
      }
    });
  });

  $(".blacklist").on("click", function() {
    $(this).remove();
  });


  $("#blacklist-btn").on("click", function() {
    let word = $("#blacklist-input").val();
    if (word.length > 0) {
      $("#blacklist").append(`<li class="blacklist" data-word="${word}"><i class="fas fa-minus-circle delete"></i></li>`);
      $("#blacklist-input").val("");
      $(".blacklist").on("click", function() {
        $(this).remove();
      });
    }
  });

  $("#new-input-btn").on('click', function() {
    if (!currently_generating) {
      let current_story = $("#current-story").text();
      let new_data = $("#new-input").val();
      let story_json = JSON.stringify({
        input: new_data,
        context: current_story
      });
      currently_generating = true;
      $.ajax({
        data: story_json,
        type: 'POST',
        url: "/generate",
        timeout: 0,
        contentType: 'application/json;charset=UTF-8',
        beforeSend: (function() {
          $("#continue-text").hide();
          $(".circle").show();
          $("#new-input-btn").addClass("disabled");
        }),
        success: function(response) {
          $("#continue-text").show();
          $(".circle").hide();
          $("#new-input-btn").removeClass("disabled");
          $("#new-input").val("");
          $("#current-story").html(response.context);
          currently_generating = false;
        }
      });
    }
  });

  $("#author-selection-btn").on('click', function() {
    if ($("#author-selection").css("display") == "none") {
      $(".setting-container").fadeOut(100);
      $("#author-selection").show();
    }
  });

  $("#genre-selection-btn").on('click', function() {
    if ($("#genre-selection").css("display") == "none") {
      $(".setting-container").fadeOut(100);
      $("#genre-selection").show();
    }
  });

  $("#story-setting-btn").on('click', function() {
    if ($("#general-settings").css("display") == "none") {
      $(".setting-container").fadeOut(100);
      $("#general-settings").css({
        'display': 'flex'
      });
    }
  });

  $("#hamburger .ham").on("click", function() {

    if (!$(this).hasClass("active")) {
      $("#setup").show();

    } else {
      $("#setup").hide();
    }
    $(this).toggleClass("active");
  });

  const hamburgerSetup = (function() {
    // $(this).addClass("active");
    // $(this).classList.toggle('active');
  });
  hamburgerSetup();

  // Creativity skew
  const MAX_SKEW = -35;
  const SKEW_STEP = MAX_SKEW/100;
  $("#creativity").css("transform", `skew(${SKEW_STEP*75}deg)`);
  $("#temperature").on("change", function() {
    let creativeVal = $(this).val()*100;
    $("#creativity").css("transform", `skew(${SKEW_STEP*creativeVal}deg)`);
  });

});
