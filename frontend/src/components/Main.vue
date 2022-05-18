<template>
  <div class="area">
    <!-- <el-container> -->
    <!-- <h2 id="header">Moral Story Generation System</h2> -->
    <!-- <el-main> -->
    <span style="margin-bottom: 5px">请选择 Moral 种类</span>
    <el-select v-model="moral" placeholder="请选择 Moral 种类">
      <el-option
        v-for="item in options"
        :key="item.value"
        :label="item.label"
        :value="item.value"
      >
      </el-option>
    </el-select>

    <el-card class="box-card-1" shadow="always">
      <div slot="header" class="clearfix">
        <span>Moral Story Generation System</span>
        <el-button
          style="float: right"
          size="small"
          @click="get_data"
          type="primary"
          icon="el-icon-reading"
          round
          >生成</el-button
        >
        <!-- <el-button type="primary" @click="get_data" icon="el-icon-reading" round>
      开始生成
            </el-button> -->
      </div>
      <div v-loading="loading">
        <el-row>
          <el-input
            type="textarea"
            :autosize="{ minRows: 10, maxRows: 20 }"
            placeholder="请输入内容"
            v-model="textarea"
            clearable
            maxlength="500"
            show-word-limit="true"
            style="border-radius: 30px"
          >
          </el-input>
        </el-row>
      </div>
    </el-card>

    <transition name="el-zoom-in-top">
      <el-card class="box-card" shadow="always" v-show="show">
        <div slot="header" class="clearfix">
          <span>系统推荐走向(也可以自己在输入框自定义输入)</span>
        </div>
        <!-- <el-skeleton :rows="4" /> -->

        <div v-for="o in cands" :key="o" class="text item">
          <el-button type="text" @click="recordtext($event)">{{
            o
          }}</el-button>
        </div>
      </el-card>
    </transition>
    <!-- </el-main> -->
    <!-- </el-container> -->
  </div>
</template>

<script>
export default {
  name: "Main",
  // 注意，这里是一个 data 方法，数据是通过 return 的形式返回的
  data() {
    return {
      textarea: "",
      moral: "",
      show: false,
      loading: false,
      options: [
        {
          value: "江山易改，本性难移。",
          label: "江山易改，本性难移。",
        },
        {
          value: "团结就是力量。",
          label: "团结就是力量。",
        },
        {
          value: "一定要爱护卫生。",
          label: "一定要爱护卫生。",
        },
        {
          value: "有志者，事竟成。",
          label: "有志者，事竟成。",
        },
        {
          value: "要坚持，不放弃。",
          label: "要坚持，不放弃。",
        },
      ],
    cands:['这是段落1','这是段落2','这是段落3','这是段落4'],
    };
  },
  methods: {
    get_data() {
      this.loading = true;
      this.show = false;
      // this.textarea = this.moral;
      let param = {
        'moral': this.moral,
        'context': this.textarea,
      };

      this.axios.post("/story", param).then((res) => {
        this.loading = false;
        this.cands = res.data.cand_list;
        this.show = true;
      });

    },

    recordtext(event) {
      let val = event.currentTarget.innerHTML.match(
        /(<span>=?)(\S*)(?=<\/span>)/
      )[2];
      this.textarea = this.textarea + val;
      this.show = false;
    },
  },
};
</script>

<style>
.area {
  display: flex;
  flex-direction: column;
  align-content: space-around;
  align-items: center;
}

.input {
  background-color: #fcfcfc;
  margin-top: 20px;
  height: 250px;
  width: 50%;
}

.text {
  font-size: 12px;
}

.el-button--text {
  color: #7f8c8d;
  /* background-color: #20B2AA;
  border-color: #20B2AA; */
}

.el-textarea__inner {
  font-family: "Microsoft";
  font-size: 15px;
}

.item {
  margin-bottom: 1px;
}

.clearfix:before,
.clearfix:after {
  display: table;
  content: "";
}
.clearfix:after {
  clear: both;
}

.box-card-1 {
  background-color: #fcfcfc;
  width: 600px;
  margin-top: 30px;
}

.box-card {
  background-color: #fcfcfc;
  width: 900px;
  height: 250px;
  margin-top: 15px;
}
</style>