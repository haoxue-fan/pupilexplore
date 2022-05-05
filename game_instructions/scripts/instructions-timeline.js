/**
 * Method to turn a number into a capatilized letter.
 * Used to set up the comprehension questions. Ensures that "A"'s
 * char code is 0, when it is usually set as 1.
 *
 * @param {Integer} num_letter The charcter number from 0-64 to use.
 */
function to_letter(num_letter) {
  return String.fromCharCode(num_letter + 64 + 1);
}

$(document).ready(function() {
  $("#templates").hide();

  const stim_info = {
    imgs: [
      "images/1_white_fixation.png",
      "images/2_white_fixation_options.png",
      "images/3_green_fixation_options.png",
      "images/4_red_fixation_options.png"
    ],
    train: {
      //TODO: to be changed
      num_blocks: 0,
      num_trials_per_block: 0
    },
    test: {
      //TODO: to be changed
      num_blocks: 0,
      num_trials_per_block: 0
    }
  };

  const instruction_pages = [
    $("#instructions-1").html(),
    $("#instructions-2").html(),
    sprintf($("#instructions-3").html(), stim_info.imgs[1]),
    sprintf(
      $("#instructions-4").html(),
      stim_info.test.num_blocks,
      stim_info.test.num_trials_per_block
    ),
    sprintf($("#instructions-5").html(), stim_info.imgs[0]),
    sprintf($("#instructions-6").html(), stim_info.imgs[1]),
    sprintf($("#instructions-7").html(), stim_info.imgs[2]),
    $("#instructions-8").html(),
    sprintf($("#instructions-9").html(), stim_info.imgs[3]),
    $("#instructions-10").html(),
    $("#instructions-11").html(),
    $("#instructions-12").html(),
    $("#instructions-13").html(),
    $("#instructions-14").html()
  ];

  const comprehension_check_info = {
    num_correct: 0,
    num_loops: 0,
    questions: [
      {
        prompt: "How do you play a slot machine?",
        options: [
          "I click my preferred choice on the screen.",
          "I press the '1' or '2' keys on my keyboard to indicate my choice of \
          the first or second option.",
          "I press the left or right arrow keys on my keyboard to indicate my \
          choice of the left or right option.",
          "I press the 'Q' or 'P' keys on my keyboard to indicate my choice of \
          the left or right option, respectively."
        ],
        expected_indx: 2,
        name: "choice"
      },
      {
        prompt: "What is the difference between Safe and Risky machines?",
        options: [
          "Playing a Safe machine always leads to winning more coins \
                than a Risky machine.",
          "Playing a Safe machine always leads to winning less coins \
                than a Risky machine.",
          "Playing a Safe machine always leads to the same outcome. \
                Playing a Risky machine leads to a different outcome each \
                time.",
          "Playing a Safe machine always lead to winning coins. Playing \
                a Risky machine may lead to winning or losing coins."
        ],
        expected_indx: 2,
        name: "difference"
      },
      {
        prompt:
          "Which of the following statements about receiving a bonus is \
              correct?",
        options: [
          "Whether I win a bonus only depends on the total coins I \
                collect in the game.",
          "Whether I win a bonus only depends on the speed I make \
                choices.",
          "However I do in the task, the bonus is unchanged."
        ],
        expected_indx: 0,
        name: "bonus"
      },
      {
        prompt:
          "Which of the following statement about behavior during the \
          experiment is correct?",
        options: [
          "I should keep my chin placed on the chinrest at all times.",
          "I should look at the cross in the middle as much as possible, \
          especially after you have made a selection.",
          "I should avoid excessive or prolonged blinking.",
          "All of the above."
        ],
        expected_indx: 3,
        name: "behavior"
      }
    ]
  };

  const preload = {
    type: "preload",
    auto_preload: true,
    message:
      "Please wait while the experiment loads. This may take a few \
        minutes.",
    error_message:
      "The experiment failed to load. Please contact the \
        researcher.",
    images: stim_info.imgs,
    on_success: function(file) {
      console.log("File loaded: ", file);
    },
    on_error: function(file) {
      console.log("Error loading file: ", file);
    }
  };

  /**
   * Format all the instructions used throughout the experiment.
   *
   * @param {Array<html>} pages An array of html pages to display.
   * @returns {Object<instructions>} A JsPych instructions object containing the
   * given pages
   * @static
   */
  function format_instructions(pages) {
    return {
      type: "instructions",
      pages: pages,
      show_clickable_nav: true,
      allow_backward: true,
      show_page_number: true,
      css_classes: ["absolute-center"]
    };
  }

  /**
   * Create and return the complete comprehension check timeline.
   *
   * Using the given questions, this randomizes the question ordering,
   * formats each multiple choice question, and includes a feedback screen.
   *
   * Global variable changes:
   * (1) comprehension_check_info (num_correct and num_loops) are incremented
   *
   * @param {Array<Map>} questions An array of questions where each question is
   * formatted as such: {
   * prompt: {String},
   * options: {Array<String>},
   * expected_indx: {Integer},
   * name: {String}
   * }
   * @returns {Array<Object<quiz-multi-choice, instructions>>}
   * comprension_check_timeline the full array with all the formatted questions
   * (JsPych quiz-multi-choice plugin) and feedback screen (JsPych
   * instructions plugin)
   */
  function comprehension_check(questions) {
    let comprehension_check_timeline = [];

    // Randomize the question ordering so as to vary the comprehension check
    // when repeated.
    const shuffled_question_indxs = jsPsych.randomization.shuffle(
      Array.from(Array(questions.length), (_, i) => i)
    );

    // Create and add all the multiple choice questions to the timeline.
    for (
      let num_question = 0;
      num_question < shuffled_question_indxs.length;
      ++num_question
    ) {
      let question = questions[shuffled_question_indxs[num_question]];

      comprehension_check_timeline.push({
        type: "quiz-multi-choice",
        prompt: `<h4>${num_question + 1}. ${question.prompt}</h4>`,
        options: function() {
          // Formatting every option to mimic a multiple choice question
          return question.options.map(function(option, num_option) {
            return `(${to_letter(num_option)}) ${option}`;
          });
        },
        expected: function() {
          let expected_indx = question.expected_indx;
          return `(${to_letter(expected_indx)}) ${
            question.options[expected_indx]
          }`;
        },
        name: question.name,
        on_finish: function(data) {
          if (data.correct) {
            ++comprehension_check_info.num_correct;
          }
        },
        css_classes: ["absolute-center"]
      });
    }

    // Include a feedback screen
    comprehension_check_timeline.push({
      type: "instructions",
      pages: [
        function() {
          let prompt =
            comprehension_check_info.num_correct ==
            comprehension_check_info.questions.length
              ? "Great job, you passed! Now you are ready to run through some \
              practice trials."
              : "Oh no! Please press the button below to repeat the \
              instructions.";

          return sprintf(
            $("#comprehension-feedback").html(),
            comprehension_check_info.num_correct,
            comprehension_check_info.questions.length,
            prompt
          );
        }
      ],
      show_clickable_nav: true,
      allow_backward: false,
      button_label_next: "Next",
      css_classes: ["absolute-center"]
    });

    return comprehension_check_timeline;
  }

  const task_introduction = {
    timeline: [format_instructions(instruction_pages)].concat(
      comprehension_check(comprehension_check_info.questions)
    ),
    on_timeline_start: function() {
      ++comprehension_check_info.num_loops;
      comprehension_check_info.num_correct = 0;
    },
    // Repeat the timeline until the participant gets all the comprehension
    // questions correct.
    loop_function: function() {
      return (
        comprehension_check_info.num_correct !=
        comprehension_check_info.questions.length
      );
    },
    on_finish: function(data) {
      jsPsych.data.write({
        num_comprehension_loops: comprehension_check_info.num_loops
      });
    },
    css_classes: ["absolute-center"]
  };

  let experiment_timeline = [preload, task_introduction];

  jsPsych.init({
    timeline: experiment_timeline,
    display_element: "jspsych-display",
    on_finish: function() {
      var closing_txt = $("#closing-remarks").html();
      $("#jspsych-display").html(closing_txt);
    }
  });
});
