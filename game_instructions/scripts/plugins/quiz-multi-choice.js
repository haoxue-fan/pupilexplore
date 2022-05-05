/*
 * Example plugin template
 */

jsPsych.plugins["quiz-multi-choice"] = (function() {

  var plugin = {};

  plugin.info = {
    name: "quiz-multi-choice",
    parameters: {
      name: {
        name: jsPsych.plugins.parameterType.STRING, // BOOL, STRING, INT, FLOAT, FUNCTION, KEY, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
        default: undefined
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        default: undefined
      },
      options: {
        type: jsPsych.plugins.parameterType.OBJECT,
        default: undefined
      },
      expected: {
        type: jsPsych.plugins.parameterType.OBJECT,
        default: undefined
      }
    }
  }

  plugin.trial = function(display_element, trial) {

    // Question
    var header = sprintf('<div class = "quiz-wrapper"><h4><label for = "%s">%s</label></h4>', trial.name, trial.prompt);

    // Fill in options
    var options = '<div class = "form-group">';
    var q_template = '<div class = "form-check"> \
    <input class="form-check-input" type="radio" name="%s" id="%s-%i" value="%s"> \
    <label class="form-check-label" for="%s-%i">%s</label></div>';
    for (i=0; i<trial.options.length; i++) {
      var opt_txt = sprintf(q_template,
        trial.name,
        trial.name,
        i,
        trial.options[i],
        trial.name,
        i,
        trial.options[i] 
        );
      options = options + opt_txt;
    }
    options = options + '</div>';

    // Fill in content
    var button = '<button class = "jspsych-btn" disabled>Next ></button>';
    var content = '<form><div class = "quiz_wrapper">' + header + options + button + '</div></form></div>';
    $(display_element).html(content);

    // Enable continue button when an option is pressed
    $(display_element).find('input[type="radio"]').on('change', function(){

      $(display_element).find('button').removeAttr('disabled');
      $(display_element).find('button').unbind('click');
      $(display_element).find('button').one('click', function(e){
        e.preventDefault();

        var ans = $(display_element).find('input[type="radio"]:checked').val();
        var trial_data = {
          answer: ans,
          correct: ans == trial.expected
        };

        jsPsych.finishTrial(trial_data);
      });

      $(display_element).find('input[type="radio"]').unbind('click');
    });

    }

  return plugin;
})();
