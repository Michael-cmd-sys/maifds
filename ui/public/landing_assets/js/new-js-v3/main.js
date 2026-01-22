(function ($) {
    "use strict";

// Preloader
$(".preloader").delay(1600).fadeOut("slow");



// Sticky Menu
$(window).on( 'scroll', function() {
    var scroll = $(window).scrollTop();
    if (scroll >= 10) {
        $(".header-menu-area").addClass("sticky");
    } else {
        $(".header-menu-area").removeClass("sticky");
    }
});


// Nice select
$(document).ready(function() {
  $('select').niceSelect();
});



// Magnific Popup video
$('.popup-youtube').magnificPopup({
    disableOn: 700,
    type: 'iframe',
    removalDelay: 160,
    fixedContentPos: false
});

// Magnific Popup gallery
$('.popup-gallery').magnificPopup({
    delegate: 'a', // child items selector, by clicking on it popup will open
    gallery: {
        enabled: true
    },
    type: 'image'
    // other options
});
        
// Mobile menu
$('.hamburger').on( 'click', function (event) {
    $(this).toggleClass('h-active');
    $('.main-nav').toggleClass('slidenav');
});
$('.header-home .main-nav ul li  a').on( 'click', function (event) {
    $('.hamburger').removeClass('h-active');
    $('.main-nav').removeClass('slidenav');
});

$(".main-nav .fl").on('click', function(event) {
    var $fl = $(this);
    $(this).parent().siblings().find('.sub-menu').slideUp();
    $(this).parent().siblings().find('.fl').addClass('flaticon-plus').text("+");
    if($fl.hasClass('flaticon-plus')){
        $fl.removeClass('flaticon-plus').addClass('flaticon-minus').text("-");
    }else{
        $fl.removeClass('flaticon-minus').addClass('flaticon-plus').text("+");
    }
    $fl.next(".sub-menu").slideToggle();
});


  
// Client Slide

$('.client-slide-wrap').owlCarousel({
  items: 6,
  loop: true,
  smartSpeed: 1500,
  autoplay: true,
  dots: false,
  margin:24,
  nav: false,
  navText : ["<i class='ts ts-1 bx bxs-left-arrow-alt'></i>","<i class='ts ts-2 bx bxs-right-arrow-alt' ></i>"], 
  responsive:{
      0:{
          items:2
      },
      480:{
          items:3
      },

      768:{
          items:4
      },
      992:{
          items:5
      },
      1200:{
          items:6
      },
      1400:{
          items:6
      }

  } 
});

// Client Slide End


  
// Testimonial Slide

$('.testimonial-slide-wrap').owlCarousel({
  items: 3,
  loop: true,
  smartSpeed: 1500,
  autoplay: false,
  dots: false,
  margin:74,
  stagePadding: 40,
  nav: true,
  navText : ["<i class='ts ts-1 bi bi-arrow-left'></i>","<i class='ts ts-2 bi bi-arrow-right' ></i>"], 
  responsive:{
      0:{
          items:1,
          stagePadding: 0,
          margin:24
      },
      480:{
          items:1,
          stagePadding: 0,
          margin:24
      },

      768:{
          items:1,
          stagePadding: 0,
          margin:24
      },
      992:{
          items:2
      },
      1200:{
          items:3
      },
      1400:{
          items:3
      }

  } 
});

// Testimonial Slide End


  
// Testimonial Slide

$('.testimonial-slide-consultant').owlCarousel({
  items: 2,
  loop: true,
  smartSpeed: 1500,
  autoplay: false,
  dots: false,
  margin:74,
  stagePadding: 40,
  nav: true,
  navText : ["<i class='ts ts-1 bi bi-arrow-left'></i>","<i class='ts ts-2 bi bi-arrow-right' ></i>"], 
  responsive:{
      0:{
          items:1,
          stagePadding: 0,
          margin:24
      },
      480:{
          items:1,
          stagePadding: 0,
          margin:24
      },

      768:{
          items:1,
          stagePadding: 0,
          margin:24
      },
      992:{
          items:2
      },
      1200:{
          items:2
      },
      1400:{
          items:2
      }

  } 
});

// Testimonial Slide End

  
// Recent Work Slide

$('.recent-work-wrap').owlCarousel({
  items: 4,
  loop: true,
  smartSpeed: 1500,
  autoplay: false,
  dots: false,
  margin:24,
  nav: true,
  navText : ["<i class='ts ts-1 bi bi-arrow-left'></i>","<i class='ts ts-2 bi bi-arrow-right' ></i>"], 
  responsive:{
      0:{
          items:1,
          stagePadding: 0,
          margin:24
      },
      480:{
          items:1,
          stagePadding: 0,
          margin:24
      },

      768:{
          items:2,
          stagePadding: 0,
          margin:24
      },
      992:{
          items:3
      },
      1200:{
          items:3
      },
      1400:{
          items:4
      }

  } 
});

// Recent Work Slide End

  
// Hero Agency Slide

$('.hero-slide-wrap').owlCarousel({
  items: 1,
  loop: true,
  smartSpeed: 1500,
  autoplay: true,
  autoplayTimeout:8000,
  dots: false,
  animateOut: 'fadeOut',
  center: true,
  margin:0,
  nav: true,
  navText : ["<i class='ts ts-1 bi bi-arrow-left'></i>","<i class='ts ts-2 bi bi-arrow-right' ></i>"], 
  responsive:{
      0:{
          items:1
      },
      480:{
          items:1
      },

      768:{
          items:1
      },
      992:{
          items:1
      },
      1200:{
          items:1
      },
      1400:{
          items:1
      }

  } 
});

// Hero Agency Slide End



// Showcase Slide

$('.showcase-slide-wrap').owlCarousel({
  items: 3.5,
  loop: true,
  smartSpeed: 1500,
  autoplay: false,
  dots: false,
  margin:24,
  nav: true,
  navText : ["<i class='ts ts-1 bi bi-arrow-left'></i>","<i class='ts ts-2 bi bi-arrow-right' ></i>"], 
  responsive:{
      0:{
          items:1,
          stagePadding: 0,
          margin:24
      },
      480:{
          items:1,
          stagePadding: 0,
          margin:24
      },

      768:{
          items:2,
          stagePadding: 0,
          margin:24
      },
      992:{
          items:2
      },
      1200:{
          items:3
      },
      1400:{
          items:3.5
      }

  } 
});

// Showcase Slide End



// Bottom to top start
$(document).ready(function () {
  $(window).scroll(function () {
    if ($(this).scrollTop() > 100) {
      $('#scroll-top').fadeIn();
    } else {
      $('#scroll-top').fadeOut();
    }
  });
  $('#scroll-top').on( 'click', function () {
    $("html, body").animate({ scrollTop: 0 }, 600);
    return false;
  });
});
// Bottom to top End

//wow Animation
jQuery(window).on( 'load', function(){

  new WOW().init();
  window.wow = new WOW(
    {
    boxClass:     'wow',      // default
    animateClass: 'animated', // default
    offset:       0,          // default
    mobile:       true,       // default
    live:         true,        // default
    offset: 100,
    callback:function(box){},
    scrollContainer:null,
    resetAnimation:true
    }
  )
  window.wow.init();
         
});
//wow Animation End


// Odometer Counter
$(".counter-item").each(function () {
  $(this).isInViewport(function (status) {
  if (status === "entered") {
      for (var i = 0; i < document.querySelectorAll(".odometer").length; i++) {
      var el = document.querySelectorAll('.odometer')[i];
      el.innerHTML = el.getAttribute("data-odometer-final");
    }
  }
  });
});


// Search Start
const searchToggle = document.querySelector(".search-icon");
  searchToggle.addEventListener("click" , () =>{
  searchToggle.classList.toggle("active");
})
// Search End


// Pricing Toggle

var checkBox = document.querySelectorAll("#checbox")

for(let i = 0; i < checkBox.length; i++){
  checkBox[i].addEventListener("click", ()=>{
    var text1 = document.querySelectorAll(".text1")
    var text2 = document.querySelectorAll(".text2")
    
    if(checkBox[i].checked == true){
      console.log(text2)
      text1.forEach((e)=>{
        e.style.display = "block";
      })
      text2.forEach((e)=>{
        e.style.display = "none";
      })
    } else if (checkBox[i].checked == false) {
      text1.forEach((e)=>{
        e.style.display = "none";
      })
      text2.forEach((e)=>{
        e.style.display = "block";
      })
    }

  })
}

// Pricing Toggle End


// Hover Active Start
const containers = document.querySelectorAll('.container-wrap');

containers.forEach(f => f.addEventListener('mouseenter', function() {
  containers.forEach(e => {
    var div = e.querySelector('.service-wrap');
    div.classList.add('show');
    div.classList.remove('active');
  })
  this.querySelector('div').classList.add('active')
}))

// Hover Active End


// Hover Active Showcase Start
const show_box = document.querySelectorAll('.showcase-box-wrap');

show_box.forEach(f => f.addEventListener('mouseenter', function() {
  show_box.forEach(e => {
    var div = e.querySelector('.showcase-box');
    div.classList.add('show');
    div.classList.remove('active');
  })
  this.querySelector('div').classList.add('active')
}))

// Hover Active Showcase End


// Menu Toggle button sidebar

var toggleIcon = document.querySelectorAll('.menu-toggle-btn i')
var closeIcon = document.querySelectorAll('.cross-icon')
var searchWrap = document.querySelectorAll('.menu-toggle-btn-full-shape')

toggleIcon.forEach((element)=>{
    element.addEventListener('click', ()=>{
        document.querySelectorAll('.menu-toggle-btn-full-shape').forEach((el)=>{
            el.classList.add('show-sidebar')
        })
    })
})

closeIcon.forEach((element)=>{
    element.addEventListener('click', ()=>{
        document.querySelectorAll('.menu-toggle-btn-full-shape').forEach((el)=>{
            el.classList.remove('show-sidebar')
        })
    })
})

window.onclick = function(event){
  // Menu Toggle button sidebar
  searchWrap.forEach((el)=>{
    if(event.target == el){
      el.classList.remove('show-sidebar')
    }
  })
}

// Menu Toggle button sidebar end



}(jQuery)); 
