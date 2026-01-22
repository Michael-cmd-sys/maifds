/*-----------------------------------------------------------------

Template Name:  Marketi - Digital Marketing Agency HTML Template
Author:  wprealizer
Author URI: https://themeforest.net/user/https://themeforest.net/user/wprealizer/portfolio/portfolio
Developer: Kawser Ahmed Roni
Version: 1.0.0
Description: Marketi - Digital Marketing Agency HTML5 Template

-------------------------------------------------------------------
CSS TABLE OF CONTENTS
-------------------------------------------------------------------

01. preloader
02. header
03. swiper slider
04. animated text with swiper slider
05. shop products count
06. image src change
07. hide & show a div
08. maenMenu 
09. add class & remove class
10. magnificPopup
11. back to top
12. data backgrund
13. coundown by click
14. remove products
15. wow animation
17. meanmenu 
16. parallax 

------------------------------------------------------------------*/

(function ($) {
  ("use strict");

  // Preloader area start here ***
  var windowOn = $(window);
  windowOn.on("load", function () {
    $("#loading").fadeOut(500);
  });
  // Preloader area end here ***

  // Header area start here ***
  // Mobile menu
  $(".header-area nav").meanmenu();

  // Menu Fixed
  var fixed_top = $(".header-area");
  $(window).on("scroll", function () {
    if ($(this).scrollTop() > 50) {
      fixed_top.addClass("menu-fixed animated fadeInDown");
    } else {
      fixed_top.removeClass("menu-fixed fadeInDown");
    }
  });
  // Header area end here ***

  // Team slider area start here ***
  var swiper = new Swiper(".team__slider", {
    loop: "true",
    spaceBetween: 40,
    speed: 400,
    pagination: {
      el: ".team__dot",
      clickable: true,
    },
    navigation: {
      nextEl: ".team__arry-next",
      prevEl: ".team__arry-prev",
    },
    breakpoints: {
      1299: {
        slidesPerView: 4,
      },
      991: {
        slidesPerView: 3,
      },
      575: {
        slidesPerView: 2,
      },
    },
  });
  // Team slider area end here ***

  // Testimonial slider area start here ***
  var swiper = new Swiper(".testimonial__slider", {
    loop: "true",
    spaceBetween: 30,
    speed: 400,
    autoplay: {
      delay: 4000,
      disableOnInteraction: false,
    },
    pagination: {
      el: ".testimonial__dot",
      clickable: true,
    },
    navigation: {
      nextEl: ".testimonial__arry-next",
      prevEl: ".testimonial__arry-prev",
    },
  });

  var swiper = new Swiper(".testimonial-two__slider", {
    loop: "true",
    spaceBetween: 30,
    speed: 400,
    pagination: {
      el: ".testimonial-two__dot",
      type: "fraction",
    },
    navigation: {
      nextEl: ".testimonial-two__arry-next",
      prevEl: ".testimonial-two__arry-prev",
    },
  });

  // Store all dependencies in variables
  const testimonialThreeSlider = document.querySelector(
    ".testimonial-three__slider"
  );
  const testimonialThreePagination = document.querySelector(
    ".testimonial-three__pagination"
  );

  if (testimonialThreeSlider && testimonialThreePagination) {
    // Initialize Swiper with main pagination
    const testimonialSlider = new Swiper(testimonialThreeSlider, {
      slidesPerView: 1,
      loop: true,
      spaceBetween: 30,

      pagination: {
        el: testimonialThreePagination,
        clickable: true,
      },

      breakpoints: {
        768: {
          slidesPerView: 2,
        },
        992: {
          slidesPerView: 1,
        },
        1300: {
          slidesPerView: 1.42,
        },
      },

      slideToClickedSlide: true,
      speed: 3000,
      autoplay: {
        delay: 1,
        disableOnInteraction: false,
      },
    });
  }

  const testimonialFourSlider = document.querySelector(
    ".testimonial-four__slider"
  );
  const testimonialFourPagination = document.querySelector(
    ".testimonial-four__pagination"
  );

  if (testimonialFourSlider && testimonialFourPagination) {
    const swiper = new Swiper(testimonialFourSlider, {
      loop: true,
      slidesPerView: 1,
      spaceBetween: 30,
      pagination: {
        el: testimonialFourPagination,
        clickable: true,
      },
      autoplay: {
        delay: 3000,
        disableOnInteraction: false,
      },
    });
  }
  // Testimonial slider area end here ***

  // Service slider area start here ***
  var swiper = new Swiper(".service__slider", {
    loop: "true",
    spaceBetween: 24,
    speed: 300,
    navigation: {
      nextEl: ".service__arry-next",
      prevEl: ".service__arry-prev",
    },
    breakpoints: {
      1200: {
        slidesPerView: 4,
      },
      992: {
        slidesPerView: 3,
      },
      575: {
        slidesPerView: 2,
      },
    },
  });
  // Service slider area end here ***

  // Service three slider area start here ***
  const serviceThreeEl = document.querySelector(".service-three__slider");
  const serviceThreeElPagination = document.querySelector(
    ".service-three__pagination"
  );

  if (serviceThreeEl && serviceThreeElPagination) {
    const swiper = new Swiper(serviceThreeEl, {
      loop: true,
      slidesPerView: 1,
      spaceBetween: 30,
      pagination: {
        el: serviceThreeElPagination,
        clickable: true,
      },
      breakpoints: {
        1200: {
          slidesPerView: 3,
        },
        768: {
          slidesPerView: 2,
        },
      },
    });
  }
  // Service three slider area end here ***

  // Brand slider area start here ***
  var swiper = new Swiper(".brand__slider", {
    loop: true,
    spaceBetween: 24,
    allowTouchMove: false,
    disableOnInteraction: true,
    speed: 3000,
    autoplay: {
      delay: 1,
    },
    breakpoints: {
      1200: {
        slidesPerView: 5,
      },
      992: {
        slidesPerView: 4,
      },
      575: {
        slidesPerView: 2,
      },
      320: {
        slidesPerView: 2,
      },
    },
  });
  // Brand slider area end here ***

  // Marque slider area start here ***
  var swiper = new Swiper(".marque-three__slider", {
    loop: true,
    allowTouchMove: false,
    disableOnInteraction: true,
    speed: 3000,
    slidesPerView: "auto",
    autoplay: {
      delay: 1,
    },
    // breakpoints: {
    //   1200: {
    //   },
    //   992: {
    //     slidesPerView: 4,
    //   },
    //   575: {
    //     slidesPerView: 2,
    //   },
    //   320: {
    //     slidesPerView: 2,
    //   },
    // },
  });
  // Marque slider area end here ***

  // Blog slider area start here ***
  var swiper = new Swiper(".blog__slider", {
    loop: "true",
    spaceBetween: 40,
    speed: 400,
    pagination: {
      el: ".blog__dot",
      clickable: true,
    },
    navigation: {
      nextEl: ".blog__arry-next",
      prevEl: ".blog__arry-prev",
    },
    breakpoints: {
      575: {
        slidesPerView: 2,
      },
    },
  });
  // Blog slider area end here ***

  // Faq slider area start here ***
  var swiper = new Swiper(".faq__slider", {
    loop: "true",
    spaceBetween: 80,
    speed: 400,
    navigation: {
      nextEl: ".faq__arry-next",
      prevEl: ".faq__arry-prev",
    },
    breakpoints: {
      768: {
        slidesPerView: 2,
      },
    },
  });
  // Faq slider area end here ***

  // Portfolio Four area start here ***
  const portfolioFourSlide = document.querySelector(".portfolio-four__slider");
  if (portfolioFourSlide) {
    const swiper = new Swiper(portfolioFourSlide, {
      loop: true,
      slidesPerView: 1,
      spaceBetween: 10,
      speed: 2500,
      // speed: 400,
      autoplay: {
        delay: 1,
        disableOnInteraction: false,
      },
      breakpoints: {
        576: {
          slidesPerView: 2,
          spaceBetween: 10,
        },
        768: {
          slidesPerView: 3,
          spaceBetween: 10,
        },
        1024: {
          slidesPerView: 3.8,
          spaceBetween: 20,
          centeredSlides: true,
        },
        1200: {
          slidesPerView: 4.5,
          spaceBetween: 25,
          centeredSlides: true,
        },
      },
    });
  }

  // Portfolio Four area end here ***

  // Active Class control area start here ***
  function setupHoverActive(
    hoverSelector,
    targetSelector = hoverSelector,
    activeClass = "active"
  ) {
    const hoverItems = document.querySelectorAll(hoverSelector);
    const targetItems = document.querySelectorAll(targetSelector);

    // Set the first item active initially
    if (targetItems.length > 0) {
      targetItems.forEach((item) => item.classList.remove(activeClass));
      targetItems[0].classList.add(activeClass);
    }

    // Add hover listener
    hoverItems.forEach((hoverItem, index) => {
      hoverItem.addEventListener("mouseenter", () => {
        targetItems.forEach((item) => item.classList.remove(activeClass));
        if (targetItems[index]) {
          targetItems[index].classList.add(activeClass);
        }
      });
    });
  }
  setupHoverActive(".process-three__item");
  setupHoverActive(".team-three__item");

  // Active Class control area end here ***

  // Mouse move paralax area end here ***
  if ($(window).width() > 780) {
    $(".paralax__animation").mousemove(function (e) {
      $("[data-depth]").each(function () {
        var depth = $(this).data("depth");
        var amountMovedX = (e.pageX * -depth) / 4;
        var amountMovedY = (e.pageY * -depth) / 4;

        $(this).css({
          transform:
            "translate3d(" + amountMovedX + "px," + amountMovedY + "px, 0)",
        });
      });
    });
  }
  // Mouse move paralax area end here ***

  // Hide & show by clicks js area start here ***
  // $(document).on("click", "#openButton", function () {
  //   $("#targetElement").removeClass("sidebar__hide");
  // });
  // $(document).on("click", "#closeButton", function () {
  //   $("#targetElement").addClass("sidebar__hide");
  // });
  $(document).on("click", "#openButton", function () {
    $("#targetElement").removeClass("sidebar__hide");
  });

  $(document).on("click", "#closeButton", function () {
    $("#targetElement").addClass("sidebar__hide");
  });

  $(document).on("click", ".sidebar__overlay", function () {
    const sidebar = $("#targetElement");
    if (!sidebar.hasClass("sidebar__hide")) {
      sidebar.addClass("sidebar__hide");
    }
  });
  // Hide & show by clicks js area end here ***

  // Background image date area start here ***
  $("[data-background").each(function () {
    $(this).css(
      "background-image",
      "url( " + $(this).attr("data-background") + "  )"
    );
  });
  // Background image date area end here ***

  // Video popup area start here ***
  $(".video-popup").magnificPopup({
    type: "iframe",
    iframe: {
      markup:
        '<div class="mfp-iframe-scaler">' +
        '<div class="mfp-close"></div>' +
        '<iframe class="mfp-iframe" frameborder="0" allowfullscreen></iframe>' +
        "</div>",

      patterns: {
        youtube: {
          index: "youtube.com/",

          id: "v=",

          src: "https://www.youtube.com/embed/%id%?autoplay=1",
        },
        vimeo: {
          index: "vimeo.com/",
          id: "/",
          src: "//player.vimeo.com/video/%id%?autoplay=1",
        },
        gmaps: {
          index: "//maps.google.",
          src: "%id%&output=embed",
        },
      },

      srcAction: "iframe_src",
    },
  });
  // Video popup area end here ***

  // Map popup area start here ***
  $(".map-popup").magnificPopup({
    disableOn: 700,
    type: "iframe",
    mainClass: "mfp-fade",
    removalDelay: 160,
    preloader: false,
    fixedContentPos: false,
  });
  // Map popup area end here ***

  // Counter up area start here ***
  $(".count").counterUp({
    delay: 30,
    time: 1000,
  });
  $(".progress-count").counterUp({
    delay: 20,
    time: 1000,
  });
  // Counter up area end here ***

  // Nice seclect area start here ***
  $(document).ready(function () {
    $("select").niceSelect();
  });
  // Nice seclect area end here ***

  // Back to top btn area start here ***
  $(window).scroll(function () {
    if ($(this).scrollTop() > 20) {
      $(".btn-backToTop").addClass("show");
    } else {
      $(".btn-backToTop").removeClass("show");
    }
  });
  $(".btn-backToTop").click(function () {
    $("html, body").animate({ scrollTop: 0 }, 800);
    return false;
  });
  // Back to top btn area end here ***

  // updateAchievements area start here ***
  // function updateAchievements() {
  //   const achievements = document.querySelectorAll(".achieve-progress");

  //   if (achievements.length > 0) {
  //     achievements.forEach((progressEl) => {
  //       const value = progressEl.getAttribute("data-achieve");

  //       if (value) {
  //         // Set the width of the progress bar
  //         progressEl.style.width = `${value}%`;

  //         // Set the text inside .skill-count
  //         const countEl = progressEl.querySelector(".skill-count");
  //         if (countEl) {
  //           countEl.textContent = `${value}%`;
  //         }
  //       }
  //     });
  //   }
  // }
  // updateAchievements();
  function animateAchievementsOnView() {
    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const el = entry.target;
            const value = el.getAttribute("data-achieve");
            if (value) {
              el.style.width = `${value}%`;
              const countEl = el.querySelector(".skill-count");
              if (countEl) countEl.textContent = `${value}%`;
            }
            obs.unobserve(el); // Animate only once
          }
        });
      },
      { threshold: 0.3 }
    );

    document
      .querySelectorAll(".achieve-progress")
      .forEach((el) => observer.observe(el));
  }

  animateAchievementsOnView();

  // updateAchievements area end here ***

  // setupPortfolioHoverSync area start here ***
  function setupPortfolioHoverSync() {
    // Declare dependencies
    const contentItems = document.querySelectorAll(".portfolio-three__content");
    const thumbItems = document.querySelectorAll(
      ".portfolio-three__thumbs .portfolio-thumb"
    );

    if (
      contentItems.length === 0 ||
      thumbItems.length === 0 ||
      contentItems.length !== thumbItems.length
    ) {
      return;
    }

    let activeIndex = 0;

    function activateItem(index) {
      contentItems.forEach((item, i) => {
        item.classList.toggle("active", i === index);
      });
      thumbItems.forEach((item, i) => {
        item.classList.toggle("active", i === index);
      });
      activeIndex = index;
    }

    // Attach hover events
    function bindHoverEvents() {
      contentItems.forEach((item, index) => {
        item.addEventListener("mouseenter", () => {
          activateItem(index);
        });
      });
    }

    activateItem(0);
    bindHoverEvents();
  }

  setupPortfolioHoverSync();
  // setupPortfolioHoverSync area end here ***

  // setupSearchBoxToggle area start here ***
  const setupSearchBoxToggle = () => {
    const searchBoxArea = document.querySelector(".searchBox-area");
    const searchCloseBtn = searchBoxArea?.querySelector(".searchClose");
    const searchBtns = document.querySelectorAll(".search-btn");
    const searchForm = searchBoxArea?.querySelector(".SearchBox__form");

    if (
      !searchBoxArea ||
      !searchCloseBtn ||
      searchBtns.length === 0 ||
      !searchForm
    )
      return;

    searchBtns.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        searchBoxArea.classList.add("show");
      });
    });

    searchForm.addEventListener("click", (e) => e.stopPropagation());

    searchBoxArea.addEventListener("click", () => {
      searchBoxArea.classList.remove("show");
    });

    searchCloseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      searchBoxArea.classList.remove("show");
    });
  };

  setupSearchBoxToggle();
  // setupSearchBoxToggle area end here ***

  // setCurrentYear area start here ***
  const setCurrentYear = (selector = ".current-year") => {
    const yearSpan = document.querySelector(selector);
    if (yearSpan) {
      yearSpan.textContent = new Date().getFullYear();
    }
  };
  setCurrentYear();
  // setCurrentYear area end here ***

  // setCurrentYear area start here ***
  const initServiceThumbSync = () => {
    const thumbImages = document.querySelectorAll(".thumb-image");
    const serviceLists = document.querySelectorAll(".service-list");

    // Run only if both groups exist and lengths match reasonably
    if (!thumbImages.length || !serviceLists.length) return;

    function clearClasses() {
      thumbImages.forEach((el) => el.classList.remove("active", "prev"));
      serviceLists.forEach((el) => el.classList.remove("active"));
    }

    function activateIndex(index) {
      clearClasses();
      if (thumbImages[index]) thumbImages[index].classList.add("active");
      if (serviceLists[index]) serviceLists[index].classList.add("active");
      if (thumbImages[index - 1]) thumbImages[index - 1].classList.add("prev");
    }

    // Initial state
    activateIndex(0);

    // Bind hover events
    serviceLists.forEach((item, index) => {
      item.addEventListener("mouseenter", () => activateIndex(index));
    });
  };
  initServiceThumbSync();
  // setCurrentYear area end here ***

  // initializeBannerFourSliders area start here ***
  function initializeBannerFourSliders() {
    // Store dependencies
    const swiperLib = window.Swiper;
    const sliderEl = document.querySelector(".banner-four__slider");
    const contentSliderEl = document.querySelector(
      ".banner-four__content-slider"
    );
    const paginationEl = document.querySelector(
      ".banner-four__slider-pagination"
    );

    // Check if all dependencies are available
    if (!swiperLib || !sliderEl || !contentSliderEl || !paginationEl) {
      return;
    }

    // Initialize sliders
    const bannerImageSlider = new Swiper(sliderEl, {
      slidesPerView: 1,
      loop: true,
      spaceBetween: 40,
      speed: 3000,
      autoplay: {
        delay: 500,
        disableOnInteraction: false,
      },

      pagination: {
        el: paginationEl,
        clickable: true,
      },
    });

    const bannerContentSlider = new Swiper(contentSliderEl, {
      slidesPerView: 1,
      loop: true,
      spaceBetween: 40,
      speed: 3000,
      autoplay: {
        delay: 500,
        disableOnInteraction: false,
      },

      pagination: {
        el: paginationEl,
        clickable: true,
      },
    });

    // Synchronize sliders
    bannerImageSlider.on("slideChange", () => {
      if (bannerContentSlider.realIndex !== bannerImageSlider.realIndex) {
        bannerContentSlider.slideToLoop(bannerImageSlider.realIndex);
      }
    });

    bannerContentSlider.on("slideChange", () => {
      if (bannerImageSlider.realIndex !== bannerContentSlider.realIndex) {
        bannerImageSlider.slideToLoop(bannerContentSlider.realIndex);
      }
    });
  }
  initializeBannerFourSliders();

  // initializeBannerFourSliders area end here ***

  // initializeBannerFourSliders area end here ***
  const initializeExpertiseFourSlider = () => {
    const SwiperClass = window.Swiper;
    const sliderEl = document.querySelector(".expertise-four__slider");
    const prevBtn = document.querySelector(
      ".expertise-four__slider-pagination .prev-slide"
    );
    const nextBtn = document.querySelector(
      ".expertise-four__slider-pagination .next-slide"
    );

    // Dependency check
    if (!SwiperClass || !sliderEl || !prevBtn || !nextBtn) {
      return;
    }

    nextBtn.classList.add("active");

    // Initialize Swiper
    const expertiseSlider = new SwiperClass(sliderEl, {
      slidesPerView: 1,
      loop: true,
      speed: 2000,
      autoplay: {
        delay: 1500,
        disableOnInteraction: false,
      },
      breakpoints: {
        768: {
          slidesPerView: 2,
        },
        1300: {
          slidesPerView: 3,
        },
      },
    });

    // Navigation click handlers
    prevBtn.addEventListener("click", () => {
      expertiseSlider.slidePrev();
      prevBtn.classList.add("active");
      nextBtn.classList.remove("active");
    });

    nextBtn.addEventListener("click", () => {
      expertiseSlider.slideNext();
      nextBtn.classList.add("active");
      prevBtn.classList.remove("active");
    });
  };
  initializeExpertiseFourSlider();

  // initializeBannerFourSliders area end here ***

  // WOW Animatin area start here ***
  new WOW().init();
  // WOW Animatin area start here ***
})(jQuery);
